# Local Large Language Model

The course uses GPT models through OpenRouter and issues every student an API
key, which is stored in `.env`.

Here I use open-weight language models to explore how well local models perform
on the assignments.

> [!NOTE]
> To do the experiment, don't forget to set a separate `CS329Z_CACHE_DIR` in
> `.env` for each configuration.

## Baseline

Qwen3.5-4B is the first model that works for Problem 1 (implemented at commit
`be6ab5d`), and here is the command to start the server.

```bash
docker run --rm -it \
    -p 9931:9931 \
    --gpus all \
    -v ~/.cache/huggingface:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda13-v1.0.0-b12359-b1b770d \
    -m /models/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-UD-Q4_K_XL.gguf \
    --parallel 12 \
    --kv-unified-per-slot 6144 \
    --host 0.0.0.0 \
    --port 9931 \
    --reasoning-budget 512
```

It took 107.52s, with 55 of 60 correct classifications (91.7% accuracy).

Setting `--reasoning-budget 512` resolved the truncation failures in this
experiment. Without a reasoning cap, some requests spent a long time generating
reasoning and reached a length limit before producing a final answer. The
wrapper then raised `OutputTruncated`. Although the exception reported zero
reasoning tokens, the captured responses contained long `reasoning_content`
fields; the wrapper uses zero when the server omits the corresponding usage
metadata.

The JSON response has the following structure (unrelated fileds are omitted;
empty strings are placeholders):

```json
{
    "choices": [
        {
            "finish_reason": "",
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": ""
            }
        }
    ]
}
```

In the truncated responses I inspected, `finish_reason` was `length`, `content`
was empty, and `reasoning_content` contained a long reasoning trace. One complete
JSON response body was 17,892 bytes. An excerpt follows.

```
... (truncated)
*   Wait, does it require action? "Go to... now".
*   But the definition of Ignore includes "broad announcements".
*   So I will stick with Ignore.

*   Wait, one nuance: "Automated notice requiring this recipient to act can be normal or urgent".
*   This implies if it requires action, it *could* be normal or urgent.
*   But "Ignore" also lists "broad announcements".
*   If it's a broad announcement, it's Ignore.
*   If it's a specific task, it's Normal/Urgent.
*   This is a broad announcement.
*   So Ignore.

*   Final check on reason sentence: "It is a broad company-wide announcement/newsletter promoting an event, which does not require immediate critical action."
*   Is it one sentence? Yes.
*   Does it match the category? Yes.

*   Wait, I should check if "Normal" is better.
```

> [!NOTE]
> The content above was captured with [mitmproxy](https://github.com/mitmproxy/mitmproxy).

For responses where `finish_reason` is `stop`, Qwen can return content like `normal|This
email is a legitimate FYI regarding market trends that does not require immediate
action or a specific deadline.`.

## Flash Attention `on`

```bash
docker run --rm -it \
    -p 9931:9931 \
    --gpus all \
    -v ~/.cache/huggingface:/models \
    ghcr.io/ggml-org/llama.cpp:server-cuda13-v1.0.0-b12359-b1b770d \
    -m /models/hub/models--unsloth--Qwen3.5-4B-GGUF/snapshots/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-UD-Q4_K_XL.gguf \
    --parallel 12 \
    --kv-unified-per-slot 6144 \
    --host 0.0.0.0 \
    --port 9931 \
    --reasoning-budget 512 \
    --flash-attn on
```

With the same prompt (commit `be6ab5d`), it takes 109.50s, with 51 of 60 correct
classifications (85%).

> [!NOTE]
> One reply included reasoning text and a closing `</think>` tag before the
> final answer. The original parser returned `unknown` because it didn't
> handle this case, and it led to 50/60 correct classifications (83.3%).
> After updating the parser, I re-evaluated the cached replies and obtained
> 51/60.

Observations:

- Verbose server logs confirmed that flash attention was enabled automatically
  in the baseline. Explicitly setting produced similar runtime.
- Both runs used the same prompt and had flash attention enabled. The second
  run chieved lower accuracy: 51/60 vs 55/60. The cause of this difference
  remains undetermined.

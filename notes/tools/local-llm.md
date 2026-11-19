# Local Large Language Model

The course uses GPT models through OpenRouter and issues every student an API
key, which is stored in `.env`.

Here I use open-weight language models to explore how well local models perform
on the assignments.

> [!NOTE]
> To do the experiment, don't forget to set a separate `CS329Z_CACHE_DIR` in
> `.env` for each configuration.

> [!IMPORTANT]
> `llama.cpp:server-cuda13` runs on an underclocked RTX 3060 Ti with the following
> command to avoid overheating:
>
> ```bash
> sudo nvidia-smi -i 0 -pm 1 && \
> sudo nvidia-smi -i 0 -lgc 210,1500 && \
> sudo nvidia-smi -i 0 -pl 150
> ```

## The first baseline

Qwen3.5-4B was the first model that works for Problem 1 (implemented at commit
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

The JSON response has the following structure (unrelated fields are omitted;
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
> The content above was captured with [mitmproxy](./mitmproxy.md).

For responses where `finish_reason` is `stop`, Qwen can return content like `normal|This
email is a legitimate FYI regarding market trends that does not require immediate
action or a specific deadline.`.

### Flash Attention `on`

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
  run achieved lower accuracy: 51/60 vs 55/60. The cause of this difference
  remains undetermined.

### Flash Attention `off`

With `--flash-attn off`, it took 125.91s with 53 of 60 correct classifications
(88.3%). The runtime is 17% longer.

> [!NOTE]
> This is a quick experiment for comparison of flash attention on / off. The
> other experiments leave `--flash-attn` unset, which is `auto`.

## The second baseline

The next move is to find a setting that the model can run fast with an acceptable
score.

### `--reasoning-budget 256`

I lowered the cap to 256 tokens to see how much generation time I could save. The
first two attempts ran into output-format problems.

In the first attempt, the `content` contains repeated two asterisks and one backtick
shown as the following:

```
"content": "**`**`**`**`..."
```

With no other characters. The ending of the `reasoning_content` was:

```
... (truncated) labeled `**ADVERTISEMENTS
```

The sequence of two asterisks and one backtick happened to be the closing marker.
Then I ran the request again from mitmproxy with nothing changed. The model
returned `content` in the correct format. The `reasoning_content` was also ended
with `ADVERTISEMENTS` but with bold mark, inline code mark closed correctly.

So I ran the whole test again with a separate cache folder. This time the prompt
was slightly changed, with extra instruction before the final output:

```
Use plain text throughout, including any reasoning.
Write reasoning as brief prose sentences without headings, bullet points,
bold or italic formatting, backticks, or code vences.
```

The request for plain text still produced Markdown, and one reply continued its
analysis in `content` before giving the classification. (The `content` contained
`</think>`.)

These two cases, combined with the earlier `</think>` case, seems like the model
ignore the boundary of thinking. For the third attempt, I added an extra parameter
to `llama.cpp`, to cue the transition to the final answer:

```
--reasoning-budget-message "I must stop reasoning now and provide the final answer in the requested format."
```

This attempt finished in 64.91s, about only 60% of the baseline, and scored 46/60.

### Prompt with examples

This experiment is to compare whether it helps if examples are included. The
prompt is:

```
Classify the email's priority from the recipient's perspective, at the time the email was sent.

Choose exactly one category.

## urgent

The recipient has to act or decide, or reports a live problem that the recipient is responsible for, AND action is needed the same day, the next working day, or by a stated deadline within two working days. Working days are Monday to Friday, counted from the email's date.

Examples:

- "The bank rejected this morning's wire. I need you to approve the corrected one by 1:00 PM or we pay a late penalty."
- "The customer's CFO is threatening to cancel. Can you call him before 4:00 today?"

## normal

Legitimate mail or personal email for this recipient or for a small working group, with no deadline in the next two workign days. This includes status updates and FYIs, ordinary questions, documents for review, meeting scheduling for later dates, replies in an ongoing discussion, and personal notes from friends or family.

Examples:

- "Attached is the draft term sheet. Let me know if you have comments before our meeting the week after next."
- "Thanks for lunch yesterday. Are you still coming to the lake in August?"

## ignore

Advertising, spam, newsletters, broad announcements, or automated notifications that require no action from this recipient.

Examples:

- "This week only: 40% off all standing desks. Reply REMOVE to unsubscribe."
- "To all employees: the annual bake sale is in the lobby on Friday from 11:00 to 1:00."

## Deciding between two labels

- The wording does not decide the label. An email with "URGENT" in the subject that asks nothing of the recipient is not `urgent`. A calm email that says "I need your signature by 3:00 PM today" is `urgent`.
- A request with a deadline more than two working days away, or with no deadline, is `normal`.
- Mass mailings and advertising are `ignore` even when they use words such as "act now" or "important", because nothing is asked of this recipient in particular.
- An automated notice that asks this specific recipient to act (for example, an expense report waiting for their approval) is `normal`, or `urgent` if it states a deadline within two working days.
- A short personal or social message from someone the recipient knows is `normal`, not `ignore`.

## Output

Return exactly one line: "category|reason". The category must be urgent, normal, or ignore. The reason must be one sentence.

DO NOT add headings, markdown, or surrounding commentary.
```

For the first run, the `--reasoning-budget` was set to `512` to match the baseline,
The `--reasoning-budget-message` was kept the same, to avoid unexpected `content`.
The revised prompt scored 55/60 at `512` tokens and 46/60 at `256` tokens, matching
the earlier runs. The similar scores made me curious about what the model was doing
with its reasoning budget.

> [!NOTE]
> This is a quick experiment with Qwen3.5-2B to have a first impression how long
> it could take and what score it could get, comparing to Qwen3.5-4B
>
> When budget was set to `256` it finished in 37.49s, but with one case that did
> not follow the format, while `finish_reason` was `stop`. The content was:
>
> ```
> Subject: I heard Steve Peace mutter
>
> Body:
> Bowen brought up the FERC order on 4 RTOs issued today. She said this ratchets everything else up in the state and what they want (truncated)
> ```
>
> The parser returned `unknown` for this case and led the test to fail.
>
> Increasing the budget to `512`, and the test finished in 68.74s, with 42 of 60
> correct classifications (70%). At the same time, one CPU core was running at
> 100%, and most of the time GPU was about 60%.
>
> This 2B model is faster, but the accuracy is reduced significantly.

### Comparing the reasoning traces

The in-context example in the context didn't affect the scores of Qwen3.5-4B.
It didn't change the runtime as well. I want to know how the model thinks with
different configuration (prompt + thinking budget). The following content is
the thinking process using Qwen3.5-4B with 256 reasoning budget:

```
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Classify the email's priority from the recipient's perspective at the time of sending.
    *   Categories: `urgent`, `normal`, `ignore`.
    *   Output Format: Exactly one line: "category|reason".
    *   Constraints: No headings, markdown, or surrounding commentary.

2.  **Analyze the Email Content:**
    *   **Date:** 2001-06-01 (Saturday, June 1, 2001).
    *   **From:** cara.semperger@enron.com
    *   **To:** shift.portland@enron.com (This implies a shift team or operational group).
    *   **Subject:** SCHEDULE GOING REAL TIME FOR JUNE 1
    *   **Body:**
        *   "This schedule was not called in properly to BPA, and is therefore going real time"
        *   Lists specific BPA/GCPUD equipment numbers.
        *   "HE 0100-0600 AND 2300-2400" (ShiftI must stop reasoning now and provide the final answer in the requested format.
```

For comparison, this is the thinking with 512 budget, for the same email:

```
Thinking Process:

1.  **Analyze the Request:**
    *   Task: Classify the email's priority from the recipient's perspective.
    *   Categories: `urgent`, `normal`, `ignore`.
    *   Constraint: Choose exactly one category.
    *   Constraint: Return exactly one line: "category|reason".
    *   Constraint: The reason must be one sentence.
    *   Constraint: DO NOT add headings, markdown, or surrounding commentary.

2.  **Analyze the Email Content:**
    *   **Date:** 2001-06-01T00:31:43Z (June 1, 2001).
    *   **From:** cara.semperger@enron.com (Internal Enron employee).
    *   **To:** shift.portland@enron.com (Shift manager/worker).
    *   **Subject:** SCHEDULE GOING REAL TIME FOR JUNE 1.
    *   **Body:**
        *   "This schedule was not called in properly to BPA, and is therefore going real time:"
        *   Lists specific times (HE 0100-0600 AND 2300-2400).
        *   "Please call BPA and GCPUD prior to this schedule becoming active to insure that it flows"
        *   "SEE ME IF YOU HAVE ANY QUESTIONS"
        *   "CARA"

3.  **Evaluate against Categories:**

    *   **urgent:**
        *   *Criteria:* Recipient has to act or decide, or reports a live problem that the recipient is responsible for, AND action is needed the same day, the next working day, or by a stated deadline within two working days.
        *   *Analysis:* The email states a schedule is "going real time" (likely meaning it's active or changing unexpectedly). It instructs the recipient to "call BPA and GCPUD prior to this schedule becoming active". It also says "SEE ME IF YOU HAVE ANY QUESTIONS". This implies immediate action is required to ensure the schedule flows correctly. The context (Enron internal shift scheduling) suggests a critical operational issue. The instruction "prior to this schedule becoming active" implies immediate attention is needed.
        *   *Decision:* This looks like an operational instruction requiring immediate attention to preventI must stop reasoning now and provide the final answer in the requested format.
```

Causal language models such as Qwen generates one token at a time. Each prediction
uses the prompt and the tokens the model has already generated. This means, the
model's reasoning written becomes part of the context for the final answer.

In the replies I inspected, Qwen3.5-4B often began by restating the task and
summarizing the email before evaluating the priority categories. In a 256-token
run, the reasoning budget ran out while the model was still summarizing the email.
That recap mostly repeated information from the prompt and left little room to
decide which category fit.

Adding in-context examples left the overall score unchanged. I wondered whether
Qwen still began with much the same recap and, with so little room left to weigh
the categories, tended to reach similar decisions. The score alone didn't tell
me whether the examples changed any individual decisions.

### Experiment: Reasoning Prefill?

I wondered **whether it's possible to alter the thinking behavior**. For example
to include part of the reasoning somewhere into the system prompt or user prompt
to make the model skip the recap and focus earlier on what the recipient needed
to do and when. Codex suggested putting it in the final `assistant` message and
asking llama.cpp to continue from there:

```json
{
    "messages": [
        { "role": "system", "content": "PLACEHOLDER" },
        { "role": "user", "content": "PLACEHOLDER" },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "The task, rubric, and output format are established. I will directly evaluate the recipient's required action and deadline.\nRecipient action:"
        }
    ],
    "continue_final_message": "reasoning_content",
    "add_generation_prompt": false
}
```

Unlisted keys are not touched, `PLACEHOLDER`s are just placeholders.

Then running with `curl`, the model didn't overthink, and with 256 token cap, it
also returned the correct format.

The next move is to refactor the code, see commit `96f8ff8`. Running Qwen3.5-4B
with `--reasoning-budget 256`, using the original prompt (commit `be6ab5d`), now
the model returned 51 of 60 correct classifications in 34.99s.

Since now the reasoning section has more room for inference, the next experiment
is to replace the original prompt to the updated one with examples, this time the
model returned 58 of 60 correct classifications in 32.78s. (llama.cpp server was
rebooted before every round.)

| Configuration | Correctness | Runtime |
| :------------ | :---------- | :------ |
| reasoning budget 512 | 55/60 (91.7%) | 107.52s |
| reasoning budget 256 | 46/60 (77%) | 64.91s |
| reasoning budget 256 + prefill | 51/60 (85%) | 34.99s |
| reasoning budget 256 + prefill + examples | 58/60 (96%) | 32.78s |

The latest experiment not only got the highest correctness, but also very fast.
It took only about 30% runtime of the baseline.

```
em-06734 gold=urgent yours=normal subject=''
em-21318 gold=normal yours=urgent subject='RE: Basin Analytics move - schedule'
```

### Checking repeatablility

I restarted llama.cpp before each run, and had two more runs with reasoning
prefill, the revised prompt, and a 256-token budget:

- the 2nd run: 58/60 in 33.51s (disagreements: `em-21311`, `em-21318`)
- the 3rd run: 59/60 in 33.90s (disagreement: `em-21318`)

One thinking process for `em-21318` says:

```
The task, rubric, and output format are established. I will directly evaluate the recipient's required action and deadline.
Recipient action: Request for two more desks.
Deadline: "The seating chart is coming Thursday."
Date: 2001-08-14.
Thursday is 2001-08-16 (2 days later).
This falls within the "next working day or by a stated deadline within two working days" criteria for urgent.
Reasoning: The request involves a resource allocation (desks) with a specific deadline (Thursday) for the seating chart, requiring action within two working days.
Category: urgent.
Reason: The request for desks has a deadline of Thursday, which is within two working days of the email date.
```

The model computed the date correctly `2001-08-16`, however it misunderstood the
meaning:

> Wilhelmina, We have 24 people now, not 22. Two analysts started last week. Can
> we get two more desks? The seating chart is coming Thursday.

The original email didn't ask "can we get two more desks *by* Thursday", it was
*the seating chart* coming out Thursday.

### Testing two more deadline examples

There was a quick experiment of including two more examples:

- urgent: "Today is Monday. Could you send me the vendor shortlist by Wednesday?
  The procurement schedule comes out Wednesday."
- normal: "Today is Monday. Could you send me the vendor shortlist? The procurement
  schedule comes out Wednesday."

I added them to show when a date in an email sets a deadline for the recipient's
action. In one replay, the model returned `normal` for `em-21318`. Its explanation
still treated Thursday as the deadline for the desks and incorrectly counted it
as more than two working days away.

> normal|The request for desks is a legitimate work matter with a deadline of Thursday,
> which is more than two working days away from the email date.

The error reappeared in the full evaluation. With the two additional examples,
the three runs scored 55, 57, and 55 out of 60. I kept the earlier prompt (commit
`d186ecd`) with reasoning prefill and a 256-token budget as my second baseline
for Problem 1.

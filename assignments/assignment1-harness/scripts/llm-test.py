from cs329z_hw1.llm import LM
from cs329z_hw1.types import Message

messages: list[Message] = [
    {"role": "user", "content": "What's 1+1? Answer briefly."},
]

lm = LM(cache=False)
reply = lm(messages)

print(reply)
print(lm.usage())

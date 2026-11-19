import os
from cs329z_hw1.llm import LM
from cs329z_hw1.types import Email, LMCallable, Message

PRIORITY_PROMPT = """
Classify the email's priority from the recipient's perspective, at the time the email was sent.

Choose exactly one category.

## urgent

The recipient has to act or decide, or reports a live problem that the recipient is responsible for, AND action is needed the same day, the next working day, or by a stated deadline no more than two working days away. Working days are Monday to Friday, counted from the email's date.

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
""".strip()


def _parse_reply(reply: str) -> dict[str, str]:
    """Parse a reply that follows the requested two-line format."""
    reply = reply.rsplit("</think>", 1)[-1].strip()
    category, _, reason = reply.strip().partition("|")
    category = category.strip().lower()
    reason = reason.strip()
    if category not in ("urgent", "normal", "ignore"):
        category = "unknown"

    if not reason:
        reason = reply.strip() or "The model returned no usable explanation."
    return {
        "category": category,
        "reason": reason,
    }


def classify_priority(email: Email, lm: LMCallable) -> dict:
    email_content = (
        f"Date: {email.date}\n"
        f"From: {email.from_}\n"
        f"To: {', '.join(email.to)}\n"
        f"Subject: {email.subject}\n\n"
        f"Body:\n{email.body}"
    )

    messages: list[Message] = [
        {"role": "system", "content": PRIORITY_PROMPT},
        {"role": "user", "content": email_content},
    ]

    if isinstance(lm, LM) and os.environ.get("CS329Z_PRIORITY_PREFILL") == "1":
        reply = lm(messages, "The task, rubric, and output format are established. I will directly evaluate the recipient's required action and deadline.\nRecipient action:")
    else:
        reply = lm(messages)
    return _parse_reply(reply)

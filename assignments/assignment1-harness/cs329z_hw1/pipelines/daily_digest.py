from cs329z_hw1.types import Email, LMCallable, Message
from .priority import classify_priority

SUMMARIZE_URGENT_EMAIL_PROMPT = """
    Summarize the content in one sentence as brief as possible. The content is an
    urgent email asking the recipient to act or decide soon. The
    summary MUST contain the compan, project or site. DO NOT include
    headings, markdown, or surrounding commentary. DO NOT include
    words like urgent or urgently.
    """.strip()

def recognize_urgent_items(email: Email, lm: LMCallable) -> str:
    email_content = (
        f"Subject: {email.subject}\n\n"
        f"Body:\n{email.body}"
    )

    messages: list[Message] = [
        {"role": "system", "content": SUMMARIZE_URGENT_EMAIL_PROMPT},
        {"role": "user", "content": email_content},
    ]
    return lm(messages)

def gen_daily_digest(emails: list[Email], lm: LMCallable) -> str:
    if emails is None or len(emails) == 0:
        return "The input was an empty list of emails."

    # TODO: refactor multithread
    compound = [
        (email, classify_priority(email, lm))
        for email in emails
    ]

    # TODO: handle normal
    urgent_emails = []
    normal_emails = []

    for email, priority in compound:
        if priority["category"] == "urgent":
            urgent_emails.append(email)
        elif priority["category"] != "ignore":
            normal_emails.append(email)

    results: list[str] = []
    for x in urgent_emails:
        results.append(recognize_urgent_items(x, lm))

    content = "\n".join(results)

    return content

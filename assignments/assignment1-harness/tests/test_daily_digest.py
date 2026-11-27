"""Tests for problem (daily_digest)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import cast, TypedDict

import pytest

from copy import deepcopy
from cs329z_hw1.llm import ScriptedLM
from cs329z_hw1 import adapters
from cs329z_hw1.types import Email
from tests.conftest import eval_slice
from tests.helpers import contains, contains_any, load_fixture

MAX_WORDS = 300  # the live tests check the length of what the model wrote


def make_email(n: int, subject: str, body: str, thread: int | None = None, reply_to=None) -> Email:
    return Email.from_dict({
        "id": f"em-9{n:04d}",
        "thread_id": f"th-9{(thread or n):04d}",
        "reply_to": reply_to,
        "from": f"person{n}@cardinal.example",
        "to": ["sam.lee@cardinal.example"],
        "date": f"2001-05-14T{8 + n:02d}:00:00Z",
        "subject": subject,
        "body": body,
    })


DAY = [
    make_email(1, "Outage at the Topock compressor", "Sam, call me before 9. We need a decision."),
    make_email(2, "", "Here is our forecast\n\n "),
    make_email(3, "RE: Outage at the Topock compressor", "Any update?", thread=1, reply_to="em-90001"),
    make_email(4, "50% off toner this week only", "Click here for deals."),
    make_email(5, "", ""),
    make_email(6, "Lunch Thursday?", "Are you free?"),
]

LONG_REPLY = " ".join(f"word{i}" for i in range(500))
"""A generated string \"word0 word1 ... word499\""""

REPLIES = {
    "500_words": LONG_REPLY,
    "500_words_on_many_lines": LONG_REPLY.replace(" ", "\n"),
    "one_word": "urgent",
    "empty": "",
    "garbage": "%%% <<>> {]",
}


def check_digest(digest, context: str) -> None:
    """Only checks if `digest` is string"""
    assert isinstance(digest, str), (
        f"run_daily_digest should return a string, got {type(digest).__name__}. {context}"
    )


@pytest.mark.parametrize("name", list(REPLIES))
def test_returns_a_string_for_any_reply(name):
    """The scripted model gives the same reply to every call, including a
    500-word one and an empty one. The digest is a string, whatever its
    length: the length limit applies to what the real model writes (live
    tests), and is met by the prompt, not by cutting the text."""
    lm = ScriptedLM([REPLIES[name]], repeat_last=True)
    digest = adapters.run_daily_digest(DAY, lm)
    check_digest(digest, f"Every model call was answered with the {name!r} scripted reply.")


def test_empty_day():
    lm = ScriptedLM([LONG_REPLY], repeat_last=True)
    digest = adapters.run_daily_digest([], lm)
    check_digest(digest, "The input was an empty list of emails.")


def test_blank_subjects_and_bodies():
    blank = [make_email(n, "", "" if n % 2 else "   \n") for n in range(1, 5)]
    lm = ScriptedLM(["normal"], repeat_last=True)
    digest = adapters.run_daily_digest(blank, lm)
    check_digest(digest, "Every email had a blank subject and a blank body.")


def test_emails_are_not_modified():
    day = deepcopy(DAY)
    before = deepcopy(day)
    adapters.run_daily_digest(day, ScriptedLM(["normal"], repeat_last=True))
    assert day == before, "run_daily_digest should not change the emails it is given."


# ------------------------------------------------------------------ live ---

_digests: list[tuple[DigestDay, str]] | None = None

class UrgentItem(TypedDict):
    id: str
    must_mention: list[list[str]]
class IgnoreItem(TypedDict):
    id: str
    must_not_mention: list[str]
class DigestDay(TypedDict):
    date: str
    email_ids: list[str]
    urgent: list[UrgentItem]
    ignore: list[IgnoreItem]

def live_digests(archive: list[Email], lm) -> list[tuple[DigestDay, str]]:
    """(gold day, your digest) for each day in tests/fixtures/digest_days.json.
    Computed once and shared by the three live tests."""
    global _digests
    if _digests is None:
        days = eval_slice(cast(list[DigestDay], load_fixture("digest_days.json")))
        by_id = {email.id: email for email in archive}

        def run(day: DigestDay):
            return adapters.run_daily_digest([by_id[i] for i in day["email_ids"]], lm)

        with ThreadPoolExecutor(max_workers=5) as pool:
            _digests = list(zip(days, pool.map(run, days)))
    assert _digests is not None
    return _digests


def report(what: str, problems: list[str], n: int) -> None:
    print(f"digest {what}: {n - len({p.split(':')[0] for p in problems})} of {n} days pass")
    if problems:
        print("problems:\n" + "\n".join(problems))
    assert not problems, f"Digest {what} check failed:\n" + "\n".join(problems)


@pytest.mark.live
def test_live_length(archive: list[Email], live_lm):
    """Each digest is non-empty and at most MAX_WORDS words, as the model
    wrote it."""
    days = live_digests(archive, live_lm)
    problems = []
    for day, digest in days:
        check_digest(digest, f"Day {day['date']}.")
        words = len(digest.split())
        if not digest.strip():
            problems.append(f"{day['date']}: the digest is empty")
        elif words > MAX_WORDS:
            problems.append(f"{day['date']}: {words} words; the limit is {MAX_WORDS} (len(digest.split()))")
    report("length", problems, len(days))


@pytest.mark.live
def test_live_mentions_every_urgent_item(archive: list[Email], live_lm):
    """Each gold urgent email lists groups of aliases. The digest must
    contain at least one alias from every group."""
    days = live_digests(archive, live_lm)
    problems = []
    for day, digest in days:
        for item in day["urgent"]:
            for group in item["must_mention"]:
                if not contains_any(digest, group):
                    problems.append(
                        f"{day['date']}: urgent email {item['id']} is not covered; expected "
                        f"one of {group} in the digest.\n    digest: {digest!r}"
                    )
    report("must-mention", problems, len(days))


@pytest.mark.live
def test_live_mentions_nothing_from_ignore_pile(archive: list[Email], live_lm):
    days = live_digests(archive, live_lm)
    problems = []
    for day, digest in days:
        for item in day["ignore"]:
            for term in item["must_not_mention"]:
                if contains(digest, term):
                    problems.append(
                        f"{day['date']}: the digest contains {term!r}, which comes from "
                        f"ignore email {item['id']}.\n    digest: {digest!r}"
                    )
    report("must-not-mention", problems, len(days))

"""Course-provided data types shared by your code and our tests.

You may add your own types anywhere in the package. The ones below are the
types that cross the boundary in ``cs329z_hw1/adapters.py``, so use them as they
are (do not rename fields).

Two kinds of type appear here. A ``TypedDict`` (Email, SearchResult,
DocWindow) is a plain dict with named keys: read ``email["subject"]``, not
``email.subject``. A ``dataclass`` (ToolSpec, ToolCall, ToolResult,
AgentConfig, AgentResult, and the rest) is an object with attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypedDict


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

# Every model the tests hand you has this shape: a list of messages in, the
# reply text out. ``cs329z_hw1.llm.LM`` and ``ScriptedLM`` are both one of
# these. There is no tool-calling or structured-output feature behind it.
LMCallable = Callable[[list[Message]], str]


@dataclass
class Email:
    """One email from data/emails/emails.jsonl, as a plain dict:
    email["subject"], email["from"], email["to"]."""

    id: str
    """\"em-00001\""""
    thread_id: str
    """\"th-00001\""""
    reply_to: Optional[str]
    """id of the email this one replies to, or None"""
    from_: str
    """this field is mapped to "from" stored in a json object"""
    to: list[str]
    date: str
    """ISO 8601, UTC, e.g. "2001-05-14T23:39:00Z\""""
    subject: str
    body: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Email:
        d = dict(d)
        d["from_"] = d.pop("from")
        return cls(**d)


# ---------------------------------------------------------------- tools ----

Status = Literal["ok", "unknown_tool", "invalid_args", "tool_error", "denied"]


@dataclass
class ToolSpec:
    """Everything the registry stores about one tool."""

    name: str
    description: str
    parameters: dict  # JSON Schema object describing the arguments
    fn: Callable[..., Any]  # called as fn(**args); its return value is str()-ed
    read_only: bool = False


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    status: Status
    content: str


@dataclass
class ParsedResponse:
    """What parse_response extracts from one model reply."""

    text: str
    tool_call: Optional[ToolCall] = None
    error: Optional[str] = None


# ---------------------------------------------------------------- agent ----

# Why a call to the agent stopped.
#   "done"          the model gave a final answer
#   "max_turns"     the model-call budget for this user message ran out
#   "token_budget"  the token budget for this user message ran out
#   "malformed"     too many consecutive malformed replies
StopReason = Literal["done", "max_turns", "token_budget", "malformed"]


@dataclass
class AgentResult:
    text: str
    """what the user is shown"""
    status: StopReason
    transcript: list[dict]
    """a copy of the whole session transcript so far (see below)"""


# The transcript is an append-only list of event dicts. It is a log, not the
# model's context: compaction changes what you send to the model and never
# removes events from the transcript. Our tests read these event types:
#
#   {"type": "user",        "content": str}
#   {"type": "assistant",   "content": str}               # raw model reply
#   {"type": "tool_call",   "name": str, "args": dict}
#   {"type": "tool_result", "name": str, "status": Status, "content": str}
#
# "content" of a tool_result is exactly the ToolResult.content you showed the
# model. A reply that could not be parsed produces an "assistant" event and
# then a {"type": "error", "content": str} event holding the message you sent
# back. You may append events of other types (for example "compaction"); the
# tests ignore types they do not know.


@dataclass
class Approval:
    approved: bool
    reason: str = ""


@dataclass
class AgentConfig:
    """Settings the tests pass to run_agent_session. Your agent must honor
    every field that the problems you have implemented so far define."""

    # Problem (agent_loop)
    max_turns: int = 20
    """model calls allowed per user message"""
    token_budget: int = 400_000
    """tokens allowed per user message"""
    # Problem (compaction)
    context_budget: int = 16_000
    """max tokens in any one LM call's messages"""
    # Problem (memory)
    memory_dir: Optional[Path] = None
    """where persistent memory lives"""
    # Problem (interactive)
    mode: Literal["confirm", "auto"] = "auto"
    user: Any = None
    """a cs329z_hw1.user.UserIO, or None for no human"""
    # Problem (guardrails)
    deny_rules: tuple[str, ...] = ()
    """regexes over the run_terminal command"""
    aux_lm: Any = None
    """A second model for calls that are not the agent's own turns
    (summaries for compaction, the LLM calls inside your Part 1 tools).
    None means "use the agent's model"."""
    workspace: Optional[Path] = None
    """Terminal workspace directory; None means the sandbox default."""
    terminal_timeout: float = 10.0
    """Seconds a run_terminal command may run; pass it to sandbox.run_terminal."""


# ------------------------------------------------------------- retrieval ---


class SearchResult(TypedDict):
    doc_id: str
    title: str
    snippet: str


class DocWindow(TypedDict):
    doc_id: str
    start_line: int
    """1-based line number of the first line in `text`"""
    end_line: int
    """1-based line number of the last line in `text`"""
    total_lines: int
    text: str

@dataclass
class CardinalDoc:
    doc_id: str
    """the file name without ``.md``"""
    title: str
    """the text of the first line that starts with "# ", or the ``doc_id`` if there is no such line"""
    text: str
    """the whole file. Line numbers in this assignment are 1-based positions in ``text.splitlines()``"""

class TokenRegistry:
    def __init__(self):
        self.__len: int = 0
        self.map: dict[str, int] = {}

    def set(self, token: str) -> tuple[int, bool]:
        if token not in self.map:
            id = self.map[token] = self.__len
            self.__len += 1
            return id, True
        else:
            return self.map[token], False

    def get(self, token: str) -> int | None:
        return self.map.get(token)

    def len(self) -> int:
        return self.__len

class BM25ParsedDocs:
    def __init__(
        self,
        token_registry: TokenRegistry,
        avgdl: float,
        lens: list[int],
        length_factors: list[float],
        postings: dict[int, list[tuple[int, int]]],
    ):
        self.token_registry = token_registry
        self.avgdl = avgdl
        self.lens = lens
        self.length_factors = length_factors
        self.postings = postings

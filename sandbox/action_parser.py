"""LLM output → Action parser.

Extracts structured actions from model responses. Supports:
  - <bash>...</bash>
  - <file_edit path="...">...</file_edit>
  - <finish/>

Error-tolerant parsing: attempts to fix incomplete tags, unmatched quotes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    BASH = "bash"
    FILE_EDIT = "file_edit"
    FINISH = "finish"


@dataclass
class Action:
    type: ActionType
    content: str
    path: str | None = None

    def __repr__(self) -> str:
        if self.type == ActionType.FINISH:
            return "Action(FINISH)"
        if self.type == ActionType.BASH:
            return f"Action(BASH, {self.content[:60]}...)" if len(self.content) > 60 else f"Action(BASH, {self.content})"
        return f"Action(FILE_EDIT, path={self.path}, len={len(self.content)})"


# ── Regex patterns ──────────────────────────────────────────────────

_BASH_RE = re.compile(r"<bash>(.*?)</bash>", re.DOTALL | re.IGNORECASE)
_FILE_EDIT_RE = re.compile(
    r'<file_edit\s+path\s*=\s*["\'](.*?)["\']\s*>(.*?)</file_edit>',
    re.DOTALL | re.IGNORECASE,
)
_FINISH_RE = re.compile(r"<finish\s*/?>", re.IGNORECASE)


class ActionParser:
    """Parse model output into a structured Action."""

    def parse(self, text: str) -> Action:
        """
        Extract the first valid action from the model response.

        Priority: finish > file_edit > bash
        Returns Action(FINISH) with error note if nothing parsable.
        """
        if not text or not text.strip():
            return Action(type=ActionType.FINISH, content="empty response")

        # Try finish first (simplest)
        if _FINISH_RE.search(text):
            return Action(type=ActionType.FINISH, content="finish")

        # Try file_edit
        match = _FILE_EDIT_RE.search(text)
        if match:
            path = match.group(1).strip()
            content = match.group(2).strip()
            # Sanity: if content looks like a bash command, treat as bash instead
            if self._looks_like_command(content):
                return Action(type=ActionType.BASH, content=content)
            return Action(type=ActionType.FILE_EDIT, content=content, path=path)

        # Try bash
        match = _BASH_RE.search(text)
        if match:
            cmd = match.group(1).strip()
            if cmd:
                return Action(type=ActionType.BASH, content=cmd)

        # Error-tolerant: try to fix incomplete tags
        fixed = self._try_fix_incomplete(text)
        if fixed:
            return fixed

        # Fallback: if text looks like a command (no tags), try as bash
        if self._looks_like_command(text):
            return Action(type=ActionType.BASH, content=text.strip())

        return Action(
            type=ActionType.FINISH,
            content=f"parse_error: no valid action found in: {text[:100]}",
        )

    # ── Error-tolerant fixers ──────────────────────────────────────

    def _try_fix_incomplete(self, text: str) -> Action | None:
        """Attempt to salvage incomplete XML tags."""

        # Missing closing </bash>
        m = re.search(r"<bash>(.*?)(?:</bash>|$)", text, re.DOTALL | re.IGNORECASE)
        if m and m.group(1).strip():
            # Check if there's content until the next tag or end
            cmd = m.group(1).strip()
            # Remove trailing incomplete tags
            cmd = re.sub(r"<(?:bash|file_edit|finish).*$", "", cmd, flags=re.IGNORECASE).strip()
            if cmd and self._looks_like_command(cmd):
                return Action(type=ActionType.BASH, content=cmd)

        # Missing closing </file_edit>
        m = re.search(
            r'<file_edit\s+path\s*=\s*["\'](.*?)["\']\s*>(.*?)(?:</file_edit>|$)',
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            path = m.group(1).strip()
            content = m.group(2).strip()
            content = re.sub(r"<(?:bash|file_edit|finish).*$", "", content, flags=re.IGNORECASE).strip()
            if path and content:
                return Action(type=ActionType.FILE_EDIT, content=content, path=path)

        return None

    def _looks_like_command(self, text: str) -> bool:
        """Heuristic: does this look like a shell command (not natural language)?"""
        text = text.strip()
        # If it starts with a common command
        common_cmds = [
            "ls", "cd", "cat", "echo", "mkdir", "touch", "rm", "cp", "mv",
            "grep", "find", "sed", "awk", "python", "python3", "pip",
            "curl", "wget", "apt", "apt-get", "chmod", "chown",
            "head", "tail", "wc", "sort", "uniq", "pwd", "whoami",
            "export", "source", "env", "bash", "sh", "make", "gcc",
            "test", "[", "[[",
        ]
        first_word = text.split()[0].lower() if text.split() else ""
        return first_word in common_cmds


# ── Response parsing helpers ────────────────────────────────────────

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)


def extract_thought_and_action(text: str) -> tuple[str, Action]:
    """
    Extract thought + action from a model response.
    Returns (thought_text, Action).
    """
    parser = ActionParser()

    thought = ""
    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group(1).strip()

    action = parser.parse(text)
    return thought, action

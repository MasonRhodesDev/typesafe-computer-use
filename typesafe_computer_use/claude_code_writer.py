"""The writer's requests, sent through the Claude Code CLI (`claude -p`) instead of an HTTP API.

For a machine where Claude Code is signed in with a Claude subscription and there is no
ANTHROPIC_API_KEY: every writer call becomes one headless `claude -p` run, on the CLI's own auth.

Each run is kept to a bare model call: no tools (`--tools ""`), no MCP servers, no hooks (a user's
prompt hooks would otherwise fire on every call), no saved session, and an empty working directory
so no project CLAUDE.md is read. The request goes in as stream-json, so the screenshot the answer
step sends arrives as a real image block; the schema goes to `--json-schema`, and the validated
object comes back as the result's `structured_output`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import httpx

CLAUDE_BIN = "claude"
TIMEOUT_SECONDS = 180
WORKDIR = Path(tempfile.gettempdir()) / "jev-claude-code-writer"


class ClaudeCodeError(Exception):
    """`claude -p` failed, timed out, or answered with an error result."""


class ClaudeCodeWriter:
    """`messages.create`, the one call writer.py makes, answered by `claude -p`."""

    def __init__(self, binary: str | None = None, runner=subprocess.run):
        self.binary = binary or os.environ.get("CLICKER_CLAUDE_BIN") or CLAUDE_BIN
        if shutil.which(self.binary) is None:
            raise ValueError(f"CLICKER_WRITER_API=claude-code needs the Claude Code CLI; {self.binary!r} is not on PATH")
        self.base_url = httpx.URL("claude-code://cli")
        self.messages = SimpleNamespace(create=self._create)
        self._run = runner

    def _create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict],
        output_config=None,
        thinking=None,
        reasoning: str | None = None,
    ):
        schema = output_config["format"]["schema"] if output_config else None
        argv = [
            self.binary,
            "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",
            "--model", model,
            "--system-prompt", system,
            "--tools", "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--settings", json.dumps({"disableAllHooks": True}),
        ]  # fmt: skip
        if schema is not None:
            argv += ["--json-schema", json.dumps(schema)]
        stdin = "".join(json.dumps({"type": "user", "message": m}) + "\n" for m in messages)
        WORKDIR.mkdir(parents=True, exist_ok=True)
        try:
            proc = self._run(argv, input=stdin, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=WORKDIR)
        except subprocess.TimeoutExpired as e:
            raise ClaudeCodeError(f"claude -p took longer than {TIMEOUT_SECONDS}s") from e
        result = _result_event(proc.stdout)
        if result is None or proc.returncode != 0:
            raise ClaudeCodeError(f"claude -p exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[-400:]}")
        if result.get("is_error"):
            raise ClaudeCodeError(f"claude -p answered with an error: {str(result.get('result'))[:400]}")
        structured = result.get("structured_output")
        text = json.dumps(structured) if structured is not None else str(result.get("result") or "")
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=_usage(result.get("usage")))


def _result_event(stdout: str) -> dict | None:
    """The final `result` event of a stream-json run, or None when the run never got that far."""
    for line in reversed(stdout.splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            return event
    return None


def _usage(usage: dict | None) -> SimpleNamespace | None:
    if not usage:
        return None
    return SimpleNamespace(
        input_tokens=usage.get("input_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )

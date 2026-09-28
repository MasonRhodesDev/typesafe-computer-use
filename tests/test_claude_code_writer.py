import json
import subprocess
from types import SimpleNamespace

import pytest

from typesafe_computer_use import claude_code_writer
from typesafe_computer_use.calls import Calls, MeteredWriter
from typesafe_computer_use.claude_code_writer import ClaudeCodeWriter
from typesafe_computer_use.config import writer_api
from typesafe_computer_use.writer import WriterError, compose_url, make_writer


class FakeClaude:
    """Stands in for `subprocess.run` of `claude -p`: records the call, replies with stream-json."""

    def __init__(self, result: dict | None = None, returncode: int = 0, raises: Exception | None = None):
        self.result = result
        self.returncode = returncode
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": argv, **kwargs})
        if self.raises:
            raise self.raises
        lines = [json.dumps({"type": "system", "subtype": "init"})]
        if self.result is not None:
            lines.append(json.dumps({"type": "result", **self.result}))
        return SimpleNamespace(returncode=self.returncode, stdout="\n".join(lines) + "\n", stderr="boom")


@pytest.fixture
def on_path(monkeypatch):
    monkeypatch.setattr(claude_code_writer.shutil, "which", lambda name: f"/usr/bin/{name}")


def url_reply():
    return {
        "subtype": "success",
        "is_error": False,
        "result": "ignored when structured_output is present",
        "structured_output": {"ok": True, "url": "https://example.com", "reason": "it is the site"},
        "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 5},
    }


def test_the_env_selects_the_cli_writer(clean_env, on_path):
    clean_env.setenv("CLICKER_WRITER_API", "claude-code")
    assert writer_api() == "claude-code"
    assert isinstance(make_writer(), ClaudeCodeWriter)


def test_a_request_is_one_bare_headless_run_with_the_schema(on_path):
    fake = FakeClaude(url_reply())
    calls = Calls()
    writer = MeteredWriter(ClaudeCodeWriter(runner=fake), calls)

    assert compose_url(writer, "open example", []) == "https://example.com"

    (call,) = fake.calls
    argv = call["argv"]
    assert argv[:2] == ["claude", "-p"]
    assert argv[argv.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in argv and "--no-session-persistence" in argv
    assert json.loads(argv[argv.index("--settings") + 1]) == {"disableAllHooks": True}
    assert json.loads(argv[argv.index("--json-schema") + 1])["required"] == ["ok", "url", "reason"]
    (line,) = call["input"].splitlines()
    sent = json.loads(line)
    assert sent["type"] == "user" and sent["message"]["role"] == "user"
    assert call["cwd"] == claude_code_writer.WORKDIR


def test_an_image_block_reaches_the_cli_as_is(on_path):
    fake = FakeClaude(url_reply())
    image = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    ClaudeCodeWriter(runner=fake).messages.create(
        model="claude-sonnet-5", max_tokens=100, system="s", messages=[{"role": "user", "content": [image]}]
    )
    sent = json.loads(fake.calls[0]["input"])
    assert sent["message"]["content"][0] == image


def test_plain_text_is_used_when_there_is_no_structured_output(on_path):
    reply = {"subtype": "success", "is_error": False, "result": '{"a": 1}'}
    out = ClaudeCodeWriter(runner=FakeClaude(reply)).messages.create(model="m", max_tokens=1, system="s", messages=[])
    assert out.content[0].text == '{"a": 1}'
    assert out.usage is None


@pytest.mark.parametrize(
    "fake",
    [
        FakeClaude(None, returncode=1),
        FakeClaude({"is_error": True, "result": "not signed in"}),
        FakeClaude(raises=subprocess.TimeoutExpired("claude", 180)),
    ],
    ids=["no-result", "error-result", "timeout"],
)
def test_a_failed_run_refuses_the_step(on_path, fake):
    with pytest.raises(WriterError):
        compose_url(ClaudeCodeWriter(runner=fake), "open example", [])


def test_a_missing_cli_is_a_config_error(monkeypatch):
    monkeypatch.setattr(claude_code_writer.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="not on PATH"):
        ClaudeCodeWriter()

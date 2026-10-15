from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GatewayTarget:
    hermes: str
    host: str | None = None
    cwd: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hermes, str) or not self.hermes:
            raise ValueError("gateway.hermes must be a nonempty string")
        for name in ("hermes", "host", "cwd"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value or "\0" in value):
                raise ValueError(f"gateway.{name} must be a nonempty string")
        if self.host and self.host.startswith("-"):
            raise ValueError("gateway.host must be an SSH destination, not an option")
        if self.host and any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in self.host):
            raise ValueError("gateway.host must be a single SSH destination")
        if self.cwd is not None and not Path(self.cwd).is_absolute():
            raise ValueError("gateway.cwd must be an absolute path on the gateway host")

    def command(self, profile: str | None = None) -> list[str]:
        if self.host is None:
            command = [sys.executable, "-u", "-m", "hermes_for_tern.bridge", "--hermes", self.hermes]
            if profile is not None:
                command.extend(["--profile", profile])
            if self.cwd is not None:
                command.extend(["--cwd", self.cwd])
            return command
        source = Path(__file__).with_name("bridge.py").read_text()
        bootstrap = (
            f"namespace={{'__name__':'hermes_bridge'}};exec({source!r},namespace);"
            f"namespace['bootstrap']({self.hermes!r},{profile!r},{self.cwd!r},{source!r})"
        )
        return [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "--",
            self.host,
            shlex.join(["python3", "-u", "-c", bootstrap]),
        ]

    def describe(self) -> str:
        return (
            shlex.join([self.hermes])
            + " on "
            + (self.host or "localhost")
            + (f" in {self.cwd}" if self.cwd else "")
        )

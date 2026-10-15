"""Reversible launcher: native surfaces for interactive chat, original Hermes elsewhere."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import tern_sdk

from .gateway import GatewayTarget

MARKER = "# hermes-for-tern managed launcher"


def settings_path() -> Path:
    return (
        Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
        / "hermes-for-tern"
        / "launcher.json"
    )


def settings() -> dict:
    path = settings_path()
    return json.loads(path.read_text()) if path.exists() else {}


def original_command(config: dict) -> str:
    if config.get("original"):
        path = Path(config["original"])
        if path.is_file():
            return str(path)
        raise RuntimeError(f"Original Hermes launcher is missing: {path}")
    found = shutil.which("hermes")
    if not found:
        raise RuntimeError("Install and configure Hermes Agent first; no hermes command was found.")
    path = Path(found)
    if path.is_file() and not path.is_symlink():
        try:
            if MARKER in path.read_text():
                raise RuntimeError("Managed launcher found without its settings. Restore its saved backup.")
        except UnicodeDecodeError:
            pass
    return found


def gateway_target(original: str, config: dict, options: dict | None = None) -> GatewayTarget:
    if "backend" in config:
        raise RuntimeError(
            "Replace launcher.json backend with gateway {host, hermes, cwd}; standalone backends are no longer supported."
        )
    gateway = config.get("gateway") or {}
    if not isinstance(gateway, dict) or set(gateway) - {"host", "hermes", "cwd"}:
        raise ValueError("gateway must contain only host, hermes and cwd")
    options = options or {}
    host = options.get("gateway_host", gateway.get("host"))
    return GatewayTarget(
        options.get("gateway_hermes", gateway.get("hermes")) or ("hermes" if host else original),
        host,
        options.get("gateway_cwd", gateway.get("cwd")),
    )


def install(target: Path | None = None, gateway: GatewayTarget | None = None) -> None:
    config = settings()
    if config:
        raise RuntimeError("A launcher is already installed. Uninstall it before reinstalling.")
    original = Path(shutil.which("hermes") or "")
    if not original.is_file():
        raise RuntimeError("Install Hermes Agent first.")
    target = (target or original).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(target.name + ".before-tern")
    if backup.exists() or backup.is_symlink():
        raise RuntimeError(f"Refusing to overwrite an existing backup: {backup}")
    # Only replace the discovered launcher itself, or create a new explicitly selected PATH shim.
    if target.exists() and target != original.absolute():
        raise RuntimeError(f"Refusing to replace an unrelated file: {target}")
    replacing = target == original.absolute()
    fallback = str(backup if replacing else original.absolute())
    body = "#!/bin/sh\n" + MARKER + "\nexec " + shlex.quote(sys.executable) + ' -m hermes_for_tern "$@"\n'
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if replacing:
        target.rename(backup)  # preserves an original symlink rather than copying its target
    try:
        path.write_text(
            json.dumps(
                {
                    "target": str(target),
                    "original": fallback,
                    "replaced": replacing,
                    "gateway": {"host": gateway.host, "hermes": gateway.hermes, "cwd": gateway.cwd}
                    if gateway
                    else {},
                },
                indent=2,
            )
            + "\n"
        )
        temp = target.with_name(target.name + ".tern-tmp")
        temp.write_text(body)
        temp.chmod(0o755)
        temp.replace(target)
    except Exception:
        path.unlink(missing_ok=True)
        if replacing and backup.exists():
            backup.rename(target)
        raise
    print(f"Installed: {target}\nOriginal Hermes: {fallback}\nRun hermes inside Tern.")


def uninstall() -> None:
    config = settings()
    if not config:
        raise RuntimeError("No managed launcher is installed.")
    target = Path(config["target"])
    if not target.is_file() or MARKER not in target.read_text():
        raise RuntimeError("The launcher has changed since installation; refusing to overwrite it.")
    original = Path(config["original"])
    if config["replaced"] and not original.exists():
        raise RuntimeError("Original launcher backup is missing; leaving the managed launcher in place.")
    if config["replaced"]:
        original.replace(target)
    else:
        target.unlink()
    settings_path().unlink()
    print("Restored the original hermes command.")


def passthrough(command: str, args: list[str]) -> int:
    os.execv(command, [command, *args])
    return 0


def native_options(args: list[str]) -> dict | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", "-m")
    parser.add_argument("--provider")
    parser.add_argument("--profile", "-p")
    parser.add_argument("--tui", action="store_true")
    parser.add_argument("--demo", nargs="?", const=300.0, type=float)
    # Hermes's own ways back into a session: `--resume <id|title|latest>`, and `-c [title]`.
    parser.add_argument("--resume", "-r")
    parser.add_argument("--continue", "-c", dest="continue_last", nargs="?", const=True)
    parser.add_argument("--gateway-host")
    parser.add_argument("--gateway-hermes")
    parser.add_argument("--gateway-cwd")
    try:
        parsed, unknown = parser.parse_known_args(args)
    except SystemExit:
        return None
    if unknown:
        return None
    return {key: value for key, value in vars(parsed).items() if value is not None and key != "tui"}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args and args[0] == "--install":
            parser = argparse.ArgumentParser(description="Install a reversible hermes launcher")
            parser.add_argument("--install", action="store_true")
            parser.add_argument("--target", type=Path)
            parser.add_argument("--gateway-host")
            parser.add_argument("--gateway-hermes")
            parser.add_argument("--gateway-cwd")
            parsed = parser.parse_args(args)
            gateway = (
                GatewayTarget(parsed.gateway_hermes or "hermes", parsed.gateway_host, parsed.gateway_cwd)
                if any((parsed.gateway_host, parsed.gateway_hermes, parsed.gateway_cwd))
                else None
            )
            install(parsed.target, gateway)
            return 0
        if args == ["--uninstall"]:
            uninstall()
            return 0
        config = settings()
        original = original_command(config)
        if args == ["--doctor"]:
            print("Original Hermes:", original)
            print("Existing gateway:", gateway_target(original, config).describe())
            print("Tern SDK:", tern_sdk.wire.VERSION, "(protocol)")
            print("Installed wrapper:", config.get("target", "not installed"))
            return 0
        if args and args[0] == "--plain":
            return passthrough(original, args[1:])
        options = native_options(args)
        if options is None:
            if any(arg.startswith("--gateway-") for arg in args):
                raise RuntimeError("Gateway options apply only to native interactive chat.")
            return passthrough(original, args)
        session = tern_sdk.connect(app="hermes", version="0.1.0", features=("edit", "undo", "send"))
        if session is None:
            if options.get("demo") is not None:
                raise RuntimeError("The visual demo needs Tern. Run hermes --demo inside Tern.")
            if any(key.startswith("gateway_") for key in options):
                raise RuntimeError("Gateway overrides need the native Tern frontend. Run Hermes inside Tern.")
            return passthrough(original, args)
        required = {"col", "card", "md", "editor", "tool", "el"}
        if not required.issubset(session.caps.kinds) or not session.caps.has("dock"):
            session.close()
            if any(key.startswith("gateway_") for key in options):
                raise RuntimeError("This Tern version cannot display the native gateway frontend.")
            return passthrough(original, args)
        from .app import App

        if options.get("demo") is not None:
            duration = options["demo"]
            if not 10 <= duration <= 1800:
                session.close()
                raise ValueError("Demo duration must be between 10 and 1800 seconds.")
            command = [
                sys.executable,
                "-u",
                "-m",
                "hermes_for_tern.demo_backend",
                "--duration",
                str(duration),
            ]
        else:
            target = gateway_target(original, config, options)
            command = target.command(options.get("profile"))
            options["gateway_cwd"] = target.cwd if target.host else target.cwd or str(Path.cwd())
            options["gateway_host"] = target.host
            options["gateway_hermes"] = target.hermes
        return App(session, command, Path.cwd(), options).run()
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Hermes for Tern: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

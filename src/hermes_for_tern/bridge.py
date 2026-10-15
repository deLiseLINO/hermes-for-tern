from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path


def bootstrap(hermes: str, profile: str | None, cwd: str | None, source: str | None = None) -> None:
    try:
        descriptor = os.dup(sys.stdout.fileno())
        os.set_inheritable(descriptor, True)
        os.environ["HERMES_TERN_RPC_FD"] = str(descriptor)
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        resolved = subprocess.run(
            [hermes, "--print-runtime-command", "--module", "os"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
            check=True,
        )
        command = json.loads(resolved.stdout)
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError("Invalid runtime command")
        index = command.index("-c") + 1
        if index >= len(command):
            raise ValueError("Missing runtime bootstrap")
        if source is None:
            source = Path(__file__).read_text()
        managed = command[index] + (
            f"; namespace={{'__name__':'hermes_bridge'}}; exec({source!r},namespace);"
            f" namespace['run']({profile!r},{cwd!r})"
        )
        command[index] = (
            "import sys\ntry:\n    exec(" + repr(managed) + ")\n"
            "except Exception:\n    print('Hermes managed gateway runtime failed.',file=sys.stderr)\n"
            "    raise SystemExit(1) from None\n"
        )
        os.execv(command[0], command)
    except (OSError, ValueError, subprocess.SubprocessError):
        print(
            "Hermes gateway runtime could not be resolved. Check the configured Hermes launcher.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


class _OutputProtocol(asyncio.streams.FlowControlMixin):
    def __init__(self, loop):
        super().__init__(loop=loop)
        self.closed = loop.create_future()

    def connection_lost(self, exc):
        super().connection_lost(exc)
        if not self.closed.done():
            self.closed.set_result(None)

    def _get_close_waiter(self, stream):
        return self.closed


async def relay(output, profile: str | None, cwd: str | None) -> None:
    if profile is not None:
        from hermes_cli.profiles import resolve_profile_env
        from hermes_constants import set_hermes_home_override

        home = resolve_profile_env(profile)
        os.environ["HERMES_HOME"] = home
        set_hermes_home_override(home)
    if cwd is not None:
        os.chdir(cwd)
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_client import connect_gateway
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    discovery = await asyncio.to_thread(discover_gateway_endpoint, get_hermes_home().resolve(), timeout=5)
    if discovery.state != "ready" or discovery.endpoint is None:
        raise RuntimeError("Existing gateway is not ready")
    reader = asyncio.StreamReader(limit=8 * 1024 * 1024)
    protocol = asyncio.StreamReaderProtocol(reader)
    loop = asyncio.get_running_loop()
    pipe, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)
    try:
        output_protocol = _OutputProtocol(loop)
        transport, _ = await loop.connect_write_pipe(lambda: output_protocol, output)
        transport.set_write_buffer_limits(high=0)
        writer = asyncio.StreamWriter(transport, output_protocol, None, loop)
        try:
            async with connect_gateway(endpoint=discovery.endpoint) as client:
                # __aenter__ creates the reader without yielding; cancel before either reader can recv.
                client.reader.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await client.reader

                async def upstream():
                    while line := await reader.readline():
                        await client.websocket.send(line.decode("utf-8").removesuffix("\n"))

                async def downstream():
                    async for raw in client.websocket:
                        writer.write(raw.encode("utf-8") if isinstance(raw, str) else raw)
                        writer.write(b"\n")
                        await writer.drain()

                tasks = [
                    asyncio.create_task(upstream()),
                    asyncio.create_task(downstream()),
                    asyncio.create_task(client.websocket.wait_closed()),
                ]
                try:
                    done, _ = await asyncio.wait(
                        [*tasks, output_protocol.closed],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in done:
                        task.result()
                    if output_protocol.closed in done:
                        raise BrokenPipeError("RPC output closed")
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            # close() would wait for a paused receiver to drain buffered output.
            transport.abort()
            await writer.wait_closed()
    finally:
        pipe.close()


def run(profile: str | None, cwd: str | None) -> None:
    descriptor = os.environ.pop("HERMES_TERN_RPC_FD", None)
    output = os.fdopen(int(descriptor) if descriptor else os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        asyncio.run(relay(output, profile, cwd))
    except Exception:
        print(
            "Existing Hermes gateway connection failed. Check gateway readiness and resume to inspect any submitted turn.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    finally:
        output.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--cwd")
    args = parser.parse_args()
    bootstrap(args.hermes, args.profile, args.cwd)


if __name__ == "__main__":
    main()

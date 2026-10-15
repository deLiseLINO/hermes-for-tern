import contextlib
import fcntl
import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_for_tern.gateway import GatewayTarget


@pytest.fixture
def managed_gateway(tmp_path):
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "hermes_cli").mkdir()
    (modules / "hermes_cli" / "__init__.py").write_text("")
    (modules / "hermes_constants.py").write_text(
        "from pathlib import Path\ndef get_hermes_home(): return Path('.')\n"
    )
    (modules / "hermes_cli" / "gateway_runtime.py").write_text(
        "from types import SimpleNamespace\nimport os\n"
        "def discover_gateway_endpoint(home, timeout):\n"
        "    return SimpleNamespace(state=os.environ.get('GATEWAY_STATE','ready'),endpoint='existing')\n"
    )
    (modules / "hermes_cli" / "gateway_client.py").write_text("""import asyncio,json,os,sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
class Socket:
    def __init__(self):
        self.frames=asyncio.Queue()
        self.closed=asyncio.Event()
        if os.environ.get('FLOOD'):
            self.frames.put_nowait(json.dumps({'method':'event','params':{'text':'x' * (4 * 1024 * 1024)}}))
        else:
            self.frames.put_nowait(json.dumps({'method':'event','params':{'type':'gateway.ready'}}))
        self.peer=asyncio.create_task(self.peer_close()) if os.environ.get('PEER_CLOSE') else None
    def __aiter__(self): return self.messages()
    async def messages(self):
        while True:
            raw=await self.frames.get()
            if raw is None: return
            if os.environ.get('FLOOD'):
                asyncio.get_running_loop().call_soon(lambda: print('gateway output burst',flush=True,file=sys.stderr))
            yield raw
    async def peer_close(self):
        await asyncio.sleep(0.05)
        self.closed.set()
        self.frames.put_nowait(None)
    async def wait_closed(self): await self.closed.wait()
    async def send(self,line): pass
    async def close(self):
        self.closed.set()
        if self.peer:
            self.peer.cancel()
            await asyncio.gather(self.peer,return_exceptions=True)
@asynccontextmanager
async def connect_gateway(*,endpoint):
    if endpoint != 'existing': raise RuntimeError('unexpected endpoint')
    if os.environ.get('CONNECT_ERROR'): raise RuntimeError(os.environ['CONNECT_ERROR'])
    socket=Socket()
    async def consume():
        async for raw in socket: pass
    client=SimpleNamespace(websocket=socket,reader=asyncio.create_task(consume()))
    try: yield client
    finally:
        client.reader.cancel()
        await asyncio.gather(client.reader,return_exceptions=True)
        await socket.close()
""")
    launcher = tmp_path / "managed hermes"
    bootstrap = f"import sys;sys.path.insert(0,{str(modules)!r});print('managed bootstrap diagnostic')"
    launcher.write_text(
        f"#!{sys.executable}\nimport json\nprint(json.dumps({[sys.executable, '-I', '-c', bootstrap]!r}))\n"
    )
    launcher.chmod(0o755)
    return launcher


def launch(target, *, stdout=subprocess.PIPE, **env):
    return subprocess.Popen(
        target.command(),
        stdin=subprocess.PIPE,
        stdout=stdout,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **env},
    )


def test_gateway_peer_close_exits_even_with_idle_input_and_bootstrap_noise(managed_gateway):
    process = launch(GatewayTarget(str(managed_gateway)), PEER_CLOSE="1")
    try:
        assert json.loads(process.stdout.readline()) == {
            "method": "event",
            "params": {"type": "gateway.ready"},
        }
        process.wait(timeout=5)
        assert process.returncode == 0
        assert process.stdout.read() == ""
        assert "managed bootstrap diagnostic" in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


@pytest.mark.parametrize("termination", ["input_eof", "output_close"])
def test_idle_gateway_exits_when_a_local_pipe_closes(managed_gateway, termination):
    process = launch(GatewayTarget(str(managed_gateway)))
    try:
        assert json.loads(process.stdout.readline())["params"]["type"] == "gateway.ready"
        if termination == "input_eof":
            stdout, _ = process.communicate(timeout=5)
            assert process.returncode == 0 and stdout == ""
        else:
            process.stdout.close()
            process.stdout = None
            process.wait(timeout=5)
            assert process.returncode == 1
            assert "Existing Hermes gateway connection failed" in process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


@pytest.mark.parametrize("termination", ["peer_close", "input_eof", "output_close"])
def test_blocked_rpc_pipe_does_not_prevent_termination(managed_gateway, termination):
    read_fd, write_fd = os.pipe()
    flags = fcntl.fcntl(write_fd, fcntl.F_GETFL)
    fcntl.fcntl(write_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    try:
        with contextlib.suppress(BlockingIOError):
            while True:
                os.write(write_fd, b"x" * 4096)
        fcntl.fcntl(write_fd, fcntl.F_SETFL, flags)
        process = launch(
            GatewayTarget(str(managed_gateway)),
            stdout=write_fd,
            FLOOD="1",
            **({"PEER_CLOSE": "1"} if termination == "peer_close" else {}),
        )
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
    try:
        # This callback runs only after the relay yields with the actual stdout pipe full.
        with selectors.DefaultSelector() as selector:
            selector.register(process.stderr, selectors.EVENT_READ)
            diagnostics = b""
            while b"gateway output burst\n" not in diagnostics:
                assert selector.select(timeout=5), "gateway did not reach its output burst"
                chunk = os.read(process.stderr.fileno(), 4096)
                assert chunk, "gateway exited before offering its output burst"
                diagnostics += chunk
        if termination == "input_eof":
            process.stdin.close()
            process.stdin = None
        elif termination == "output_close":
            os.close(read_fd)
            read_fd = None
        # Do not drain stdout, including during wait: that would hide the deadlock.
        process.wait(timeout=5)
        assert process.returncode == (1 if termination == "output_close" else 0)
        stderr = diagnostics.decode() + process.stderr.read()
        assert "Traceback" not in stderr and "Task was destroyed" not in stderr
        if termination == "output_close":
            assert "Existing Hermes gateway connection failed" in stderr
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
        if read_fd is not None:
            os.close(read_fd)


@pytest.mark.parametrize(
    "failure",
    [
        {"GATEWAY_STATE": "absent"},
        {"CONNECT_ERROR": "wss://owner.invalid/api/ws?ticket=secret-ticket " * 1000},
    ],
)
def test_gateway_failure_has_bounded_diagnostics_without_secrets(managed_gateway, failure):
    process = launch(GatewayTarget(str(managed_gateway)), **failure)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 1 and stdout == ""
    assert "Existing Hermes gateway connection failed" in stderr
    assert len(stderr) < 500
    assert "Traceback" not in stderr and "secret-ticket" not in stderr and "owner.invalid" not in stderr


def test_ssh_bootstrap_preserves_quoted_launcher_and_cwd_as_data(managed_gateway, tmp_path):
    ssh = tmp_path / "ssh"
    ssh.write_text(f"#!{sys.executable}\nimport os,sys\nos.execv('/bin/sh',['sh','-c',sys.argv[-1]])\n")
    ssh.chmod(0o755)
    cwd = tmp_path / "remote ' dir; $(touch injected)"
    cwd.mkdir()
    process = launch(
        GatewayTarget(str(managed_gateway), "work", str(cwd)),
        PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
        PEER_CLOSE="1",
    )
    try:
        process.wait(timeout=5)
        stdout, stderr = process.communicate()
        assert process.returncode == 0, stderr
        assert json.loads(stdout)["params"]["type"] == "gateway.ready"
        assert not Path("injected").exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


@pytest.mark.parametrize(
    "host",
    [
        "-oProxyCommand=touch injected",
        "work -oProxyCommand=touch",
        "work\n-oProxyCommand=touch",
    ],
)
def test_gateway_rejects_ssh_option_destinations(host):
    with pytest.raises(ValueError, match="gateway.host"):
        GatewayTarget("hermes", host)


@pytest.mark.parametrize("host", [None, "work"])
@pytest.mark.parametrize("cwd", ["relative/project", "~/project"])
def test_gateway_cwd_must_be_absolute_on_the_selected_host(host, cwd):
    with pytest.raises(ValueError, match="gateway.cwd must be an absolute path"):
        GatewayTarget("hermes", host, cwd)

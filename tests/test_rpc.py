import sys

import pytest

from hermes_for_tern.rpc import Backend


def test_server_request_can_be_answered_while_submission_waits(tmp_path):
    server = tmp_path / "server.py"
    server.write_text("""import json,sys
def send(frame): print(json.dumps(frame), flush=True)
pending = None
for line in sys.stdin:
    frame = json.loads(line)
    if frame.get("method") == "prompt.submit":
        pending = frame["id"]
        send({"jsonrpc": "2.0", "id": "srq-1", "method": "clarify", "params": {"question": "Proceed?"}})
    elif frame.get("id") == "srq-1":
        if frame.get("result") != {"answer": "yes"}:
            sys.exit(9)
        send({"jsonrpc": "2.0", "id": pending, "result": {"status": "accepted", "submission_id": "turn-1"}})
""")
    backend = Backend([sys.executable, "-u", str(server)], tmp_path)
    try:
        result = backend.request("prompt.submit", {"submission_id": "turn-1", "text": "work"})
        question = backend.events.get(timeout=3)
        assert question["id"] == "srq-1" and question["method"] == "clarify"
        assert not result.done()
        backend.answer(question["id"], {"answer": "yes"})
        assert result.result(timeout=3) == {"status": "accepted", "submission_id": "turn-1"}
    finally:
        backend.close()
    assert backend.process.poll() == 0


def test_backend_crash_releases_pending_requests(tmp_path):
    backend = Backend([sys.executable, "-c", "import sys; sys.stdin.readline(); sys.exit(7)"], tmp_path)
    future = backend.request("go")
    try:
        with pytest.raises(BrokenPipeError):
            future.result(timeout=3)
    finally:
        backend.close()


def test_farewell_detaches_only_the_supplied_subscription(tmp_path):
    server = tmp_path / "server.py"
    server.write_text("""import json,sys
for line in sys.stdin:
    frame=json.loads(line)
    if frame['method'] != 'session.detach' or frame['params'] != {'session_id':'shared','subscription_id':'viewer'}:
        sys.exit(9)
    print(json.dumps({'id':frame['id'],'result':{'detached':True}}),flush=True)
""")
    backend = Backend([sys.executable, "-u", str(server)], tmp_path)
    backend.close(("session.detach", {"session_id": "shared", "subscription_id": "viewer"}))
    assert backend.process.returncode == 0

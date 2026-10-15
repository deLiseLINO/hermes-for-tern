"""Deterministic backend for native UI approval QA. No model, credentials or tools."""

import json
import sys


def send(frame):
    print(json.dumps({"jsonrpc": "2.0", **frame}), flush=True)


def event(kind, payload=None):
    send({"method": "event", "params": {"type": kind, "session_id": "qa", "payload": payload or {}}})


event("gateway.ready")
for line in sys.stdin:
    frame = json.loads(line)
    method, rid = frame.get("method"), frame.get("id")
    if method == "client.capabilities":
        send({"id": rid, "result": {"server_requests": ["clarify"]}})
    elif method == "session.create":
        send(
            {
                "id": rid,
                "result": {
                    "session_id": "qa",
                    "stored_session_id": "qa",
                    "subscription_id": "qa-viewer",
                    "revision": 0,
                    "execution_generation": 1,
                    "replay_epoch": "qa",
                    "last_sequence": 0,
                    "running": False,
                    "pending": [],
                    "prompts": [],
                    "display_messages": [],
                    "info": {"model": "UI test · no model calls"},
                },
            }
        )
    elif method == "prompt.submit":
        send(
            {"id": rid, "result": {"status": "started", "admission_id": "qa-turn", "execution_generation": 1}}
        )
        event("message.start")
        event(
            "approval.request",
            {
                "kind": "approval",
                "prompt_id": "approval-qa",
                "execution_generation": 1,
                "choices": ["once", "deny"],
                "description": "Controlled UI test. This does not execute a command.",
                "command": "printf 'approval-ui-ok\\n'",
            },
        )
    elif method == "session.interrupt":
        event("approval.settled", {"prompt_id": "approval-qa"})
        event("message.complete", {"status": "interrupted"})
        send({"id": rid, "result": {"status": "interrupted"}})
    elif method == "session.detach":
        send({"id": rid, "result": {"detached": True}})
        break
    elif method == "approval.respond":
        choice = frame["params"]["choice"]
        send({"id": rid, "result": {"prompt_id": "approval-qa", "status": "resolved"}})
        event("approval.settled", {"prompt_id": "approval-qa"})
        event("message.complete", {"status": "complete", "text": f"Approval answer received: {choice}."})
    else:
        send({"id": rid, "error": {"code": -32601, "message": "Unknown method"}})

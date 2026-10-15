import queue
import shlex
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

from tern_sdk import Key

from hermes_for_tern.app import App
from hermes_for_tern.editor import Draft
from hermes_for_tern.state import QUEUE, Attachment, Conversation, Question


class FakeBackend:
    def __init__(self):
        self.requests = []
        self.answers = []
        self.events = queue.Queue()

    def request(self, method, params):
        future = Future()
        self.requests.append((method, params, future))
        return future

    def answer(self, rid, result=None, **kwargs):
        self.answers.append((rid, result, kwargs))


def snapshot(session_id="live", stored="stored", **changes):
    return {
        "session_id": session_id,
        "stored_session_id": stored,
        "subscription_id": "viewer-" + session_id,
        "revision": 7,
        "execution_generation": 3,
        "replay_epoch": "epoch",
        "last_sequence": 0,
        "running": False,
        "pending": [],
        "prompts": [],
        "info": {},
        "display_messages": [],
        **changes,
    }


def app():
    instance = App(SimpleNamespace(caps=SimpleNamespace(cols=100)), [], Path("/project"), {})
    instance.backend = FakeBackend()
    instance.attachment = Attachment.snapshot(snapshot())
    return instance


def test_startup_schedules_session_creation_without_losing_its_future():
    instance = app()
    instance.request("client.capabilities", {"server_requests": True})
    instance.backend.requests[-1][2].set_result({"server_requests": ["approval", "clarify"]})
    instance.check_requests()
    assert instance.pending[0][0] == "session.create"
    instance.backend.requests[-1][2].set_result(snapshot())
    instance.check_requests()
    assert instance.state.session_id == "live"
    assert instance.state.ready


def test_approval_only_sends_an_offered_choice_and_waits_for_owner_settlement():
    instance = app()
    instance.state.session_id = "live"
    instance.state.event(
        "approval.request",
        {"kind": "approval", "prompt_id": "a", "execution_generation": 3, "choices": ["once", "deny"]},
    )
    instance.approve("a", "always")
    assert not instance.backend.requests and not instance.backend.answers
    instance.approve("a", "once")
    assert instance.backend.requests[-1][:2] == (
        "approval.respond",
        {
            "session_id": "live",
            "prompt_id": "a",
            "choice": "once",
            "execution_generation": 3,
        },
    )
    assert "a" in instance.state.questions
    settled(instance, {"prompt_id": "a", "status": "resolved"})
    assert not instance.state.questions


def test_multiple_clarifications_and_multiselect_answer_together():
    instance = app()
    question = Question(
        "srq-q",
        "clarify",
        {
            "questions": [
                {"qid": "a", "question": "Color?"},
                {"qid": "b", "question": "Sizes?", "multi_select": True, "choices": ["S", "M"]},
            ]
        },
    )
    instance.state.questions[question.rid] = question
    instance.answer_question(question, "Blue")
    assert not instance.backend.answers
    instance.choose(question.rid, "S")
    instance.choose(question.rid, "M")
    instance.choose(question.rid, None)
    assert instance.backend.answers == [("srq-q", {"answers": {"a": "Blue", "b": "M, S"}}, {})]


def test_unsupported_request_fails_promptly_and_busy_submit_is_held():
    instance = app()
    instance.state.session_id = "live"
    instance.server_request({"id": "srq-t", "method": "tour", "params": {"session_id": "live"}})
    assert instance.backend.answers[0][2]["error"]["code"] == -32601
    assert instance.state.rows[-1].text.startswith("This frontend does not support tour")
    instance.state.ready = True
    instance.state.begin("work")
    instance.submit("Do something")
    assert not instance.backend.requests


def pending_approval(instance, choices=("once", "session", "always", "deny")):
    instance.state.session_id = "live"
    instance.state.event(
        "approval.request",
        {"kind": "approval", "prompt_id": "a-1", "choices": list(choices), "execution_generation": 3},
    )
    return "a-1"


def test_a_pending_approval_answers_to_digits_enter_and_escape():
    for key, expected in (("3", "always"), ("enter", "once"), ("escape", "deny")):
        instance = app()
        pending_approval(instance)
        instance.input(Key(name=key))
        assert instance.backend.requests[-1][1]["choice"] == expected
        assert not instance.backend.answers


def test_a_digit_past_the_last_choice_is_ordinary_typing_and_esc_denies_the_ask():
    instance = app()
    instance.state.session_id = "live"
    instance.state.ready = True
    pending_approval(instance, choices=("once", "deny"))
    instance.input(Key(name="7", text="7"))
    assert not instance.backend.answers
    assert instance.draft.text == "7"
    instance.input(Key(name="escape"))
    assert instance.backend.requests[-1][1]["choice"] == "deny"


def test_clarification_digits_pick_a_choice_until_the_answer_draft_has_text():
    instance = app()
    instance.state.session_id = "live"
    question = Question(
        "srq-c", "clarify", {"questions": [{"qid": "a", "question": "Size?", "choices": ["S", "M"]}]}
    )
    instance.state.questions[question.rid] = question
    instance.input(Key(name="2"))
    assert instance.backend.answers == [("srq-c", {"answers": {"a": "M"}}, {})]

    instance.state.questions[question.rid] = question
    instance.answer_draft = Draft("42", 2)
    instance.input(Key(name="2", text="2"))
    assert instance.answer_draft.text == "422"
    assert len(instance.backend.answers) == 1


def test_a_message_composed_while_hermes_works_is_held_then_sent_when_the_turn_ends():
    instance = app()
    instance.state.session_id = "live"
    instance.state.ready = True
    instance.state.begin("first")
    instance.submit("and then this")
    assert instance.state.queued == ["and then this"]
    assert not [r for r in instance.backend.requests if r[0] == "prompt.submit"]

    instance.backend.events.put(
        {"method": "event", "params": {"session_id": "live", "type": "message.complete", "payload": {}}}
    )
    instance.read_backend()
    assert not instance.state.queued
    assert [r[1]["text"] for r in instance.backend.requests if r[0] == "prompt.submit"] == ["and then this"]


def test_the_queue_never_outgrows_its_limit_and_drops_the_oldest():
    state = Conversation()
    for index in range(QUEUE + 3):
        state.queue(f"note {index}")
    assert len(state.queued) == QUEUE
    assert state.queued[0] == "note 3"


def test_clear_empties_the_transcript_but_keeps_the_live_session():
    state = Conversation()
    state.session_id = "live"
    state.ready = True
    state.begin("hello")
    state.event("tool.complete", {"tool_id": "t", "name": "terminal", "result": {"output": "x"}})
    state.queue("later")
    state.clear()
    assert not state.rows and not state.tools
    # A held follow-up is pending work, not transcript: /clear must not eat what was typed.
    assert state.queued == ["later"]
    assert state.session_id == "live" and state.ready


def test_doctor_reports_the_launcher_without_leaving_the_pane():
    instance = app()
    instance.state.session_id = "live"
    instance.run_command("/doctor")
    said = " ".join(row.text for row in instance.state.rows)
    assert "project: /project" in said
    assert "session: live" in said


CATALOG = {
    "model": "claude-sonnet-4",
    "provider": "anthropic",
    "providers": [
        {
            "slug": "anthropic",
            "name": "Anthropic",
            "models": ["claude-sonnet-4", "claude-haiku-4"],
            "authenticated": True,
            "pricing": {"claude-sonnet-4": {"input": "$3", "output": "$15"}},
        },
        {
            "slug": "openai",
            "name": "OpenAI",
            "models": ["gpt-5"],
            "authenticated": False,
            "key_env": "OPENAI_API_KEY",
        },
        {"slug": "copilot", "name": "GitHub Copilot", "models": ["claude-sonnet-4"], "authenticated": True},
    ],
}


def ready(instance):
    instance.state.session_id = "live"
    instance.state.ready = True


def settled(instance, result):
    """Answer the newest request and let the loop pick the result up."""
    instance.backend.requests[-1][2].set_result(result)
    instance.check_requests()


def test_the_model_command_asks_hermes_for_its_catalog_once_and_opens_the_sheet():
    instance = app()
    ready(instance)
    instance.draft = Draft("/model", len("/model"))
    instance.submit()
    assert instance.state.sheet is not None
    assert instance.backend.requests[-1][0] == "model.options"
    settled(instance, CATALOG)
    # the sheet is already open, so a second press only refreshes it
    instance.model_event("open")
    assert [method for method, *_ in instance.backend.requests].count("model.options") == 2


def test_typing_while_the_sheet_is_open_filters_it_instead_of_the_composer():
    instance = app()
    ready(instance)
    instance.model_event("open")
    instance.sheet_key(Key(name="m", text="m"))
    assert instance.draft.text == ""
    assert instance.state.sheet.query == "m"
    for name, text in (("o", "o"), ("d", "d"), ("e", "e"), ("l", "l")):
        instance.input(Key(name=name, text=text))
    assert instance.state.sheet.query == "model"
    assert instance.draft.text == ""
    instance.input(Key(name="backspace"))
    assert instance.state.sheet.query == "mode"
    instance.input(Key(name="escape"))
    assert instance.state.sheet is None


def test_model_sheet_selection_refreshes_authority_before_mutating():
    instance = app()
    ready(instance)
    instance.state.models = CATALOG
    instance.state.info = {"model": "claude-sonnet-4", "provider": "anthropic"}
    instance.model_event("open")
    instance.confirm_model()
    settled(instance, snapshot(revision=12, execution_generation=8))
    method, params, _ = instance.backend.requests[-1]
    assert method == "session.mutate"
    assert params["payload"] == {"model": "claude-sonnet-4", "provider": "anthropic"}
    assert params["expected_revision"] == 12 and params["expected_generation"] == 8
    assert params["request_id"]


def test_an_expensive_model_preserves_the_owners_confirmation_token():
    instance = app()
    ready(instance)
    instance.model_event("open")
    instance.switch_model("anthropic/claude-sonnet-4")
    settled(instance, snapshot(revision=10, execution_generation=4))
    settled(
        instance,
        {
            "status": "confirmation_required",
            "confirm_required": True,
            "confirm": "owner-token",
            "confirm_message": "This model is expensive.",
        },
    )
    assert instance.state.sheet.confirming
    assert instance.state.rows[-1].text == "This model is expensive."
    instance.confirm_model()
    settled(instance, snapshot(revision=11, execution_generation=5))
    params = instance.backend.requests[-1][1]
    assert params["payload"] == {"model": "anthropic/claude-sonnet-4", "confirm": "owner-token"}
    assert params["expected_revision"] == 11 and params["expected_generation"] == 5


def test_a_stale_model_mutation_is_visible_and_never_retried():
    instance = app()
    ready(instance)
    instance.switch_model("gpt-5")
    settled(instance, snapshot())
    answer(instance, "session.mutate", error={"code": 4001, "message": "stale_generation"})
    assert instance.state.rows[-1].text == "stale_generation"
    assert [method for method, _, _ in instance.backend.requests].count("session.mutate") == 1


def test_a_masked_ask_is_answered_from_the_composer_and_never_shown():
    instance = app()
    instance.state.session_id = "live"
    instance.surface = SimpleNamespace(focus=lambda _: None)
    instance.server_request(
        {
            "id": "srq-s",
            "method": "secret",
            "params": {"session_id": "live", "env_var": "GITHUB_TOKEN", "prompt": "Token?"},
        }
    )
    assert instance.state.secret() is not None
    # typing is masked in the view and the real value never reaches the wire as a row
    from hermes_for_tern.views import mask

    instance.input(Key(name="g", text="g"))
    for ch in "hp_abc123":
        instance.input(Key(name=ch, text=ch))
    assert mask(instance.secret.text) == "••••••••••"
    editors = [node.wire() for node in built_nodes(instance) if node.wire().get("k") == "editor"]
    assert [editor["p"]["text"] for editor in editors] == ["••••••••••"]
    # and the answer itself appears nowhere in what the pane is sent
    shown = [node.wire().get("p", {}).get("text") for node in built_nodes(instance)]
    assert "ghp_abc123" not in [value for value in shown if isinstance(value, str)]
    instance.submit()
    assert instance.backend.answers == [("srq-s", {"value": "ghp_abc123"}, {})]
    assert not instance.state.questions
    assert instance.secret.text == ""  # the answer is not kept once it is sent


def test_escape_skips_a_masked_ask_the_way_hermes_words_it():
    instance = app()
    instance.state.session_id = "live"
    instance.surface = SimpleNamespace(focus=lambda _: None)
    instance.server_request(
        {"id": "srq-v", "method": "vault.code", "params": {"session_id": "live", "site": "bank"}}
    )
    instance.input(Key(name="escape"))
    assert instance.backend.answers == [("srq-v", {"value": ""}, {})]


def test_a_side_question_goes_to_hermes_and_its_answer_arrives_as_a_dispatch():
    instance = app()
    instance.state.session_id = "live"
    instance.state.ready = True
    instance.side_question("what is the capital of France?")
    method, params, _ = instance.backend.requests[-1]
    assert method == "prompt.btw" and params["text"] == "what is the capital of France?"
    instance.backend.events.put(
        {
            "method": "event",
            "params": {
                "session_id": "live",
                "type": "btw.complete",
                "payload": {"task_id": "t1", "question": "what is the capital of France?", "text": "Paris."},
            },
        }
    )
    instance.read_backend()
    assert [d.text for d in instance.state.dispatches] == ["what is the capital of France?"]
    assert [d.sub for d in instance.state.dispatches] == ["Paris."]


def test_a_watched_child_reports_its_tail_and_can_be_interrupted_or_steered():
    instance = app()
    instance.state.session_id = "live"
    instance.state.begin("audit")
    instance.state.event(
        "subagent.start", {"subagent_id": "a1", "goal": "check the tests", "task_index": 0, "model": "small"}
    )
    instance.watch_agent("a1")
    assert instance.state.watch is not None
    method, params, _ = instance.backend.requests[-1]
    assert method == "subagent.tail" and params["subagent_id"] == "a1"
    instance.backend.requests[-1][2].set_result(
        {"subagent_id": "a1", "available": True, "text": "reading tests…", "truncated": True}
    )
    instance.check_requests()
    assert instance.state.watch.tail == "reading tests…" and instance.state.watch.truncated

    instance.input(Key(name="r", text="r"))
    for ch in "un faster":
        instance.input(Key(name=ch, text=ch))
    assert instance.state.watch.steer == "run faster"
    instance.input(Key(name="enter"))
    method, params, _ = instance.backend.requests[-1]
    assert method == "subagent.steer" and params["text"] == "run faster"
    assert instance.state.watch.steer == ""
    instance.backend.requests[-1][2].set_result(
        {"status": "queued", "subagent_id": "a1", "text": "run faster"}
    )
    instance.check_requests()
    assert [d.text for d in instance.state.dispatches] == ["steered the child"]

    instance.interrupt_agent()
    assert instance.backend.requests[-1][0] == "subagent.interrupt"
    instance.input(Key(name="escape"))
    assert instance.state.watch is None


def test_typing_in_the_composer_is_the_watchs_line_while_a_child_is_watched():
    instance = app()
    instance.state.session_id = "live"
    instance.state.watch = None
    instance.watch_agent("a1")
    instance.draft = Draft("a thought of my own")
    instance.input(Key(name="x", text="x"))
    assert instance.state.watch.steer == "x"
    assert instance.draft.text == "a thought of my own"


def built_nodes(instance):
    from hermes_for_tern.views import view

    from pathlib import Path as _Path

    tree = view(
        instance.state,
        instance.draft,
        instance.answer_draft,
        _Path("/project"),
        lambda: None,
        lambda *a: None,
        lambda *a: None,
        lambda *a: None,
        lambda *a: None,
        secret=instance.secret,
    )
    from tern_sdk.reconcile import View

    return View.build(tree).nodes().values()


def test_rename_uses_refreshed_revision_and_owner_update():
    instance = app()
    ready(instance)
    instance.rename_session("Hunting a flaky test")
    settled(instance, snapshot(revision=19))
    method, params, _ = instance.backend.requests[-1]
    assert method == "session.mutate"
    assert params["operation"] == "rename" and params["payload"] == {"title": "Hunting a flaky test"}
    assert params["expected_revision"] == 19 and "expected_generation" not in params
    settled(
        instance,
        {
            "session_id": "live",
            "revision": 20,
            "execution_generation": 3,
            "operation": "rename",
            "title": "Hunting a flaky test",
        },
    )
    assert instance.state.title == "Hunting a flaky test"
    assert instance.attachment.revision == 20


def test_a_held_follow_up_can_be_taken_back():
    instance = app()
    instance.state.ready = True
    instance.state.begin("work")
    instance.state.queue("first")
    instance.state.queue("second")
    instance.state.unqueue(0)
    assert instance.state.queued == ["second"]
    instance.state.unqueue(5)
    assert instance.state.queued == ["second"]


def ready_app():
    instance = app()
    instance.state.session_id = "live"
    instance.state.ready = True
    return instance


def answer(instance, method, result=None, error=None):
    """Answer the newest request for `method` and let the app read it."""
    future = next(f for m, _, f in reversed(instance.backend.requests) if m == method)
    if error:
        from hermes_for_tern.rpc import RpcError

        future.set_exception(RpcError(error))
    else:
        future.set_result(result)
    instance.check_requests()


def type_text(instance, text):
    for char in text:
        instance.input(Key(name=char, text=char))


HERMES_ITEMS = {
    "items": [
        {"text": "model", "display": "/model", "meta": "Switch model (usage: /model [m])", "kind": "command"},
        {"text": "moa", "display": "/moa", "meta": "Mixture of agents", "kind": "command"},
        {"text": "compress", "display": "/compress", "meta": "Compress context", "kind": "command"},
        {"text": "new", "display": "/new", "meta": "Start a new session", "kind": "command"},
        {"text": "work", "display": "/work", "meta": "A skill", "kind": "skill"},
    ],
    "replace_from": 1,
}


def test_typing_a_slash_shows_the_list_and_each_letter_narrows_it():
    from hermes_for_tern.views import slash_rows

    instance = ready_app()
    type_text(instance, "/")
    assert instance.draft.text == "/"  # the slash stays in the composer
    # this pane's commands show at once, before Hermes answers
    assert [row["name"] for row in slash_rows(instance.state, "/")][:3] == ["/model", "/btw", "/bg"]
    assert instance.backend.requests[-1][:2] == ("complete.slash", {"session_id": "live", "text": "/"})
    answer(instance, "complete.slash", HERMES_ITEMS)
    names = [row["name"] for row in slash_rows(instance.state, "/")]
    # Hermes's own commands and skills follow; its /model is this pane's, and /new is hidden
    assert names[-3:] == ["/moa", "/compress", "/work"] and names.count("/model") == 1 and "/new" not in names
    type_text(instance, "mo")
    # until Hermes answers "/mo", its last answer is narrowed to what is typed
    assert [row["name"] for row in slash_rows(instance.state, "/mo")] == ["/model", "/moa"]
    assert instance.backend.requests[-1][1]["text"] == "/mo"


def test_the_arrows_move_tab_completes_and_enter_runs_through_hermes():
    instance = ready_app()
    type_text(instance, "/")
    answer(instance, "complete.slash", HERMES_ITEMS)
    type_text(instance, "com")
    answer(instance, "complete.slash", {"items": HERMES_ITEMS["items"][2:3], "replace_from": 1})
    instance.input(Key(name="tab"))
    assert instance.draft.text == "/compress "
    instance.draft = Draft("/com", 4)
    instance.input(Key(name="enter"))
    # not one of this pane's commands: Hermes runs it
    assert instance.backend.requests[-1][:2] == ("slash.exec", {"session_id": "live", "command": "compress"})
    answer(instance, "slash.exec", {"output": "Compressed 12 turns."})
    assert instance.state.rows[-1].kind == "notice" and instance.state.rows[-1].text == "Compressed 12 turns."


def test_escape_puts_the_list_away_until_the_draft_changes():
    from hermes_for_tern.views import slash_open

    instance = ready_app()
    type_text(instance, "/mo")
    assert slash_open(instance.state, instance.draft)
    instance.input(Key(name="escape"))
    assert not slash_open(instance.state, instance.draft)
    assert instance.draft.text == "/mo"
    type_text(instance, "d")
    assert slash_open(instance.state, instance.draft)


def test_this_panes_commands_run_here_and_ones_needing_an_argument_wait_for_it():
    instance = ready_app()
    type_text(instance, "/bt")
    instance.input(Key(name="enter"))
    assert instance.draft.text == "/btw "  # /btw needs its question: Enter completes, nothing runs
    assert not any(method == "prompt.btw" for method, _, _ in instance.backend.requests)
    instance.run_command("/cost")
    instance.run_command("/clear")
    instance.run_command("/title Flaky test hunt")  # Hermes's spelling of /name
    settled(instance, snapshot())
    assert instance.backend.requests[-1][1]["payload"] == {"title": "Flaky test hunt"}
    instance.run_command("/exit")
    assert instance.exit


def test_long_output_folds_into_a_row_and_help_adds_this_panes_commands():
    instance = ready_app()
    instance.run_command("/help")
    answer(
        instance, "slash.exec", {"output": "Available Commands\n" + "\n".join(f"/c{i}" for i in range(20))}
    )
    row = instance.state.rows[-1]
    assert row.kind == "command" and row.name == "/help" and row.collapsed
    assert "In this pane: /model" in row.text


def test_skills_go_through_command_dispatch_and_send_what_they_expand_to():
    instance = ready_app()
    instance.run_command("/work fix the leak")
    answer(instance, "slash.exec", error={"code": 4018, "message": "skill command"})
    assert instance.backend.requests[-1][:2] == (
        "command.dispatch",
        {"session_id": "live", "name": "work", "arg": "fix the leak"},
    )
    answer(
        instance,
        "command.dispatch",
        {"type": "skill", "name": "work", "message": "SKILL BODY", "display": "/work fix the leak"},
    )
    method, params, _ = instance.backend.requests[-1]
    assert method == "prompt.submit" and params["text"] == "SKILL BODY"
    assert params["submission_id"] and params["clarify_batch"] is True
    # the transcript shows the command as typed, not the expansion written for the model
    assert instance.state.rows[-1].kind == "user" and instance.state.rows[-1].text == "/work fix the leak"


def test_hermes_refusals_and_prefills_are_shown_as_hermes_says_them():
    instance = ready_app()
    instance.run_command("/undo")
    answer(instance, "slash.exec", error={"code": 4018, "message": "pending input"})
    answer(
        instance,
        "command.dispatch",
        {"type": "prefill", "message": "my last message", "notice": "Undid one turn."},
    )
    assert instance.draft.text == "my last message"
    assert instance.state.rows[-1].text == "Undid one turn."
    instance.run_command("/compress")
    answer(instance, "slash.exec", error={"code": 4009, "message": "Hermes is busy; try when the turn ends."})
    assert instance.state.rows[-1].text == "/compress: Hermes is busy; try when the turn ends."


def resuming_app(**options):
    instance = App(SimpleNamespace(caps=SimpleNamespace(cols=100)), [], Path("/project"), options)
    instance.backend = FakeBackend()
    instance.request("client.capabilities", {"server_requests": True})
    answer(instance, "client.capabilities", {})
    return instance


STORED = snapshot(
    session_id="live-2",
    stored="20261011_abc",
    info={"title": "Flaky test hunt", "model": "m"},
    display_messages=[
        {"role": "user", "text": "Why does CI fail?"},
        {"role": "assistant", "text": "Let me look."},
        {"role": "tool", "name": "terminal", "context": "uv run pytest -q"},
        {"role": "assistant", "text": "The fixture closes the pipe early."},
    ],
)


def test_resume_picks_up_the_stored_session_and_redraws_it():
    instance = resuming_app(resume="20261011_abc")
    assert instance.backend.requests[-1][:2] == (
        "session.resume",
        {"session_id": "20261011_abc", "display_projection": True},
    )
    answer(instance, "session.resume", STORED)
    state = instance.state
    assert state.session_id == "live-2" and state.stored_session_id == "20261011_abc" and state.ready
    assert [row.kind for row in state.rows] == ["user", "assistant", "tool", "assistant", "notice"]
    assert state.rows[2].target == "uv run pytest -q" and state.rows[2].status == "done"


def test_continue_asks_owner_for_latest_and_only_not_found_starts_fresh():
    instance = resuming_app(continue_last=True, profile="work")
    assert instance.backend.requests[-1][:2] == (
        "session.resume",
        {"latest": "tui", "workspace": "/project", "display_projection": True},
    )
    answer(instance, "session.resume", error={"code": 4001, "message": "not_found"})
    assert instance.backend.requests[-1][0] == "session.create"
    assert "This is a new session." in instance.state.rows[-1].text
    broken = resuming_app(continue_last=True)
    broken.backend.requests[-1][2].set_exception(BrokenPipeError("lost"))
    broken.check_requests()
    assert broken.state.failed
    assert not any(method == "session.create" for method, _, _ in broken.backend.requests)


def test_a_session_that_never_spoke_leaves_no_resume_hint():
    instance = ready_app()
    instance.state.stored_session_id = "20261011_new"
    assert instance.resume_hint() == ""


def test_one_ctrl_c_never_leaves_two_in_a_row_do():
    import time

    instance = ready_app()
    instance.input(Key(name="c", ctrl=True))
    assert not instance.exit and instance.state.exit_armed is not None
    instance.input(Key(name="c", ctrl=True))
    assert instance.exit

    # the first press stops a running turn or clears the draft, and the window closes after a while
    instance = ready_app()
    instance.state.begin("hello")
    instance.input(Key(name="c", ctrl=True))
    assert instance.backend.requests[-1][0] == "session.interrupt" and not instance.exit
    instance.state.pace(time.monotonic() + 3)
    assert instance.state.exit_armed is None
    event(instance, "message.complete", {"status": "interrupted"})
    instance.draft = Draft("half a thought", 14)
    instance.input(Key(name="c", ctrl=True))
    assert instance.draft.text == "" and not instance.exit


LISTED = {
    "sessions": [
        {"id": "20261011_now", "title": "", "preview": "", "message_count": 0, "started_at": 0},
        {"id": "20261010_ci", "title": "Flaky test hunt", "message_count": 12, "started_at": 1},
        {
            "id": "20261009_old",
            "title": "",
            "preview": "Plan the launch",
            "message_count": 4,
            "started_at": 1,
        },
    ]
}


def switching_app():
    instance = ready_app()
    instance.state.stored_session_id = "20261011_now"
    instance.state.begin("hello")
    event(instance, "message.complete", {})
    return instance


def test_resume_lists_sessions_and_picks_one_up_in_place():
    instance = switching_app()
    instance.state.info["project"] = "old project"
    instance.run_command("/resume")
    assert instance.state.sheet.kind == "sessions"
    assert {
        params["scope"] for method, params, _ in instance.backend.requests if method == "session.list"
    } == {"live", "stored"}
    answer(instance, "session.list", LISTED)
    instance.input(Key(name="enter"))  # the newest session other than this one
    assert instance.backend.requests[-1][:2] == (
        "session.resume",
        {"session_id": "20261010_ci", "display_projection": True},
    )
    answer(instance, "session.resume", STORED)
    state = instance.state
    assert state.sheet is None and state.session_id == "live-2" and state.stored_session_id == "20261011_abc"
    assert "project" not in state.info
    assert [row.kind for row in state.rows][0] == "user" and state.rows[0].text == "Why does CI fail?"
    assert "hello" not in [row.text for row in state.rows]  # the old transcript is gone
    assert instance.backend.requests[-1][:2] == (
        "session.detach",
        {"session_id": "live", "subscription_id": "viewer-live"},
    )


def test_resume_refuses_mid_turn_and_says_when_you_are_already_there():
    instance = switching_app()
    instance.state.begin("work")
    instance.run_command("/resume 20261010_ci")
    assert instance.state.rows[-1].text == "Finish or stop this turn before switching sessions."
    event(instance, "message.complete", {})
    instance.run_command("/resume 20261011_now")
    assert instance.state.rows[-1].text == "This is the session you are in."
    instance.run_command("/sessions")  # Hermes's spelling
    assert instance.state.sheet.kind == "sessions"


def test_a_failed_switch_leaves_this_session_as_it_was():
    instance = switching_app()
    instance.run_command("/resume nowhere")
    answer(instance, "session.resume", error={"code": 4001, "message": "permission_denied"})
    assert instance.state.session_id == "live" and instance.state.rows[0].text == "hello"
    assert "Could not resume 'nowhere'" in instance.state.rows[-1].text
    assert not any(method == "session.create" for method, _, _ in instance.backend.requests)


def test_a_new_session_offers_the_last_one_on_the_welcome():
    from tern_sdk.reconcile import View

    from hermes_for_tern.views import view

    instance = resuming_app()
    answer(instance, "session.create", snapshot(stored="20261011_now"))
    assert {
        params["scope"] for method, params, _ in instance.backend.requests if method == "session.list"
    } == {"live", "stored"}
    answer(instance, "session.list", LISTED)
    assert instance.state.earlier["id"] == "20261010_ci"  # not this new one, and not an empty one
    picked = []
    built = View.build(
        view(instance.state, Draft(), Draft(), Path("/p"), *[lambda *a: None] * 5, resume=picked.append)
    )
    node = built.nodes()["layer.stage.welcome.suggestions.continue"]
    assert built.nodes()["layer.stage.welcome.suggestions.continue.label"].wire()["p"]["text"] == (
        "Continue “Flaky test hunt”"
    )
    node.handlers["action:click"](None)
    assert picked == ["20261010_ci"]


def event(instance, kind, payload, **envelope):
    instance.backend.events.put(
        {"method": "event", "params": {"session_id": "live", "type": kind, "payload": payload, **envelope}}
    )
    instance.read_backend()


def test_remote_startup_uses_remote_cwd_without_gateway_options_or_shared_title():
    instance = resuming_app(
        gateway_host="work",
        gateway_cwd="/remote/project",
        gateway_hermes="/remote/bin/hermes",
        profile="work profile",
        model="m",
    )
    method, params, _ = instance.backend.requests[-1]
    assert method == "session.create"
    assert params["cwd"] == "/remote/project" and params["model"] == "m"
    assert params["request_id"]
    assert not {"title", "cols", "gateway_host", "gateway_cwd", "gateway_hermes", "profile"}.intersection(
        params
    )
    remote = resuming_app(gateway_host="work")
    assert "cwd" not in remote.backend.requests[-1][1]


def test_resume_title_fallback_only_follows_explicit_not_found():
    instance = resuming_app(resume="My session")
    answer(instance, "session.resume", error={"code": 4001, "message": "not_found"})
    assert instance.backend.requests[-1][:2] == (
        "session.resume",
        {"title": "My session", "display_projection": True},
    )
    answer(instance, "session.resume", STORED)
    assert instance.state.stored_session_id == "20261011_abc"
    broken = switching_app()
    broken.resume_session("other")
    broken.backend.requests[-1][2].set_exception(TimeoutError("no response"))
    broken.check_requests()
    assert broken.state.session_id == "live"
    assert "no response" in broken.state.rows[-1].text
    assert not any(method in ("session.create", "session.detach") for method, _, _ in broken.backend.requests)


def test_running_resume_adopts_prompts_pending_history_and_control_generation():
    instance = resuming_app(resume="live")
    answer(
        instance,
        "session.resume",
        snapshot(
            stored="live",
            running=True,
            execution_generation=9,
            pending=[
                {"admission_id": "a", "input_id": "peer-input", "text": "peer work", "status": "started"}
            ],
            prompts=[
                {
                    "kind": "clarify",
                    "prompt_id": "q",
                    "execution_generation": 9,
                    "question": "Which?",
                    "choices": ["A", "B"],
                }
            ],
        ),
    )
    assert instance.state.busy and instance.state.rows[-1].text == "peer work"
    instance.choose("q", "A")
    assert instance.backend.requests[-1][:2] == (
        "clarify.respond",
        {"session_id": "live", "prompt_id": "q", "answer": "A", "execution_generation": 9},
    )
    event(instance, "clarify.settled", {"prompt_id": "q"})
    assert not instance.state.questions
    instance.stop()
    assert instance.backend.requests[-1][1]["execution_generation"] == 9


def test_peer_pending_input_deduplicates_authored_turn_and_updates_fences():
    instance = ready_app()
    instance.submit("mine")
    submission = instance.backend.requests[-1][1]["submission_id"]
    payload = {
        "running": True,
        "revision": 15,
        "execution_generation": 8,
        "pending": [
            {"admission_id": "a", "input_id": submission, "text": "mine", "status": "started"},
            {"admission_id": "b", "input_id": "peer", "text": "theirs", "status": "queued"},
        ],
    }
    event(instance, "session.info", payload)
    event(instance, "session.info", payload)
    assert [row.text for row in instance.state.rows if row.kind == "user"] == ["mine", "theirs"]
    assert instance.attachment.revision == 15 and instance.attachment.execution_generation == 8


def test_submission_failure_and_unknown_admission_never_send_queued_followup():
    for failure in ("transport", "gap", "disconnect", "unknown"):
        instance = ready_app()
        instance.submit("first")
        instance.submit("held")
        if failure == "transport":
            instance.backend.requests[-1][2].set_exception(BrokenPipeError("lost"))
            instance.check_requests()
        elif failure == "gap":
            event(instance, "session.replay_gap", {})
        elif failure == "unknown":
            event(
                instance,
                "session.info",
                {"running": False, "pending": [{"admission_id": "a", "status": "unknown"}]},
            )
        else:
            instance.backend.events.put({"local_error": "lost"})
            instance.read_backend()
        event(instance, "session.info", {"running": False, "pending": []})
        event(instance, "message.complete", {"text": "completed"})
        assert instance.state.unknown and instance.state.queued == ["held"]
        assert not instance.state.can_retry
        assert [method for method, _, _ in instance.backend.requests].count("prompt.submit") == 1
        if failure == "disconnect":
            assert instance.state.failed and not instance.state.ready
            assert instance.state.activity == "Disconnected"
        assert any("Do not resend" in row.text for row in instance.state.rows)


def test_each_authored_submission_has_a_distinct_id_and_batch_clarification():
    instance = ready_app()
    instance.submit("first")
    event(instance, "message.complete", {})
    instance.send_command("expanded", "/skill")
    prompts = [params for method, params, _ in instance.backend.requests if method == "prompt.submit"]
    assert len({params["submission_id"] for params in prompts}) == 2
    assert all(params["clarify_batch"] is True for params in prompts)


def test_picker_deduplicates_stored_lineage_and_prefers_live_session_details():
    instance = ready_app()
    instance.open_sessions()
    answer(
        instance,
        "session.list",
        {
            "scope": "stored",
            "sessions": [
                {
                    "id": "old-row",
                    "session_id": "old-row",
                    "resolved_id": "shared",
                    "title": "stored",
                    "started_at": 1,
                    "message_count": 2,
                },
                {
                    "id": "newer",
                    "session_id": "newer",
                    "title": "newest",
                    "started_at": 9,
                    "message_count": 1,
                },
            ],
        },
    )
    future = next(
        future
        for method, params, future in instance.backend.requests
        if method == "session.list" and params["scope"] == "live"
    )
    future.set_result(
        {
            "scope": "live",
            "sessions": [
                {"id": "shared", "session_id": "shared", "title": "live", "started_at": 5, "message_count": 4}
            ],
        }
    )
    instance.check_requests()
    assert [(row["id"], row["title"]) for row in instance.state.sessions] == [
        ("newer", "newest"),
        ("shared", "live"),
    ]


def test_session_events_arriving_before_resume_result_are_applied_after_snapshot():
    instance = resuming_app(resume="live")
    event(instance, "message.delta", {"text": "already streaming"}, replay_epoch="epoch", seq=1)
    assert not instance.state.rows
    instance.backend.events.put(
        {
            "method": "event",
            "params": {
                "session_id": "live",
                "type": "message.delta",
                "payload": {"text": " later"},
                "replay_epoch": "epoch",
                "seq": 2,
            },
        }
    )
    answer(instance, "session.resume", snapshot(stored="live", running=True))
    instance.read_backend()
    assert [row.text for row in instance.state.rows if row.kind == "assistant"] == ["already streaming later"]
    assert instance.attachment.last_sequence == 2 and not instance.state.unknown


def test_model_fence_refresh_does_not_drop_streamed_reply_between_snapshots():
    instance = ready_app()
    instance.submit("work")
    event(instance, "message.delta", {"text": "first"}, replay_epoch="epoch", seq=1)
    instance.switch_model("m")
    event(instance, "message.delta", {"text": " second"}, replay_epoch="epoch", seq=2)
    instance.backend.events.put(
        {
            "method": "event",
            "params": {
                "session_id": "live",
                "type": "message.delta",
                "payload": {"text": " third"},
                "replay_epoch": "epoch",
                "seq": 3,
            },
        }
    )
    settled(instance, snapshot(last_sequence=2, running=True, revision=20, execution_generation=5))
    instance.read_backend()
    assert [row.text for row in instance.state.rows if row.kind == "assistant"] == ["first second third"]
    assert instance.attachment.last_sequence == 3 and not instance.state.unknown


def test_replayed_event_is_not_rendered_twice_and_missing_sequence_blocks_sends():
    instance = ready_app()
    event(instance, "message.delta", {"text": "once"}, replay_epoch="epoch", seq=1)
    event(instance, "message.delta", {"text": "once"}, replay_epoch="epoch", seq=1)
    assert [row.text for row in instance.state.rows if row.kind == "assistant"] == ["once"]
    event(instance, "message.delta", {"text": "lost"}, replay_epoch="epoch", seq=3)
    instance.submit("new work")
    assert instance.state.unknown
    assert not any(method == "prompt.submit" for method, _, _ in instance.backend.requests)


def test_explicit_retry_has_a_new_submission_id_after_owner_retryable_failure():
    instance = ready_app()
    instance.submit("work")
    first = instance.backend.requests[-1][1]["submission_id"]
    event(
        instance,
        "message.complete",
        {"status": "error", "text": "temporary failure", "error_surface": {"retryable": True}},
    )
    instance.retry()
    method, params, _ = instance.backend.requests[-1]
    assert method == "prompt.submit" and params["text"] == "work"
    assert params["submission_id"] != first and params["clarify_batch"] is True


def test_new_sessions_have_distinct_creation_ids_and_no_shared_title():
    instance = app()
    instance.create_session()
    instance.create_session()
    params = [params for method, params, _ in instance.backend.requests if method == "session.create"]
    assert len({item["request_id"] for item in params}) == 2
    assert all("title" not in item for item in params)


def test_idle_info_before_final_keeps_local_and_resumed_peer_followups_in_order():
    for origin in ("local", "peer"):
        if origin == "local":
            instance = ready_app()
            instance.submit("first")
            answer(instance, "prompt.submit", {"admission_id": "a", "status": "started"})
        else:
            instance = resuming_app(resume="live")
            answer(
                instance,
                "session.resume",
                snapshot(
                    running=True,
                    pending=[{"admission_id": "a", "input_id": "peer", "text": "first", "status": "started"}],
                    display_messages=[{"role": "user", "text": "first"}],
                ),
            )
        event(instance, "message.delta", {"text": "reply"})
        instance.submit("queued")
        event(instance, "session.info", {"running": False, "pending": []})
        assert instance.state.busy and instance.state.activity != "Ready"
        instance.submit("manual during gap")
        before = [
            params["text"] for method, params, _ in instance.backend.requests if method == "prompt.submit"
        ]
        assert before == (["first"] if origin == "local" else [])
        assert instance.state.queued == ["queued", "manual during gap"]
        event(instance, "message.complete", {"text": "reply done"}, admission_id="a")
        assert instance.state.queued == ["manual during gap"]
        event(instance, "message.complete", {"text": "queued reply"})
        event(instance, "message.complete", {"text": "manual reply"})
        assert [(row.kind, row.text) for row in instance.state.rows if row.kind in ("user", "assistant")] == [
            ("user", "first"),
            ("assistant", "reply done"),
            ("user", "queued"),
            ("assistant", "queued reply"),
            ("user", "manual during gap"),
            ("assistant", "manual reply"),
        ]


def test_each_admitted_turn_must_finish_before_a_local_followup_starts():
    instance = ready_app()
    instance.submit("first")
    submission = instance.backend.requests[-1][1]["submission_id"]
    event(
        instance,
        "session.info",
        {
            "running": True,
            "pending": [
                {"admission_id": "a", "input_id": submission, "text": "first", "status": "started"},
                {"admission_id": "b", "input_id": "peer", "text": "peer", "status": "queued"},
            ],
        },
    )
    instance.submit("local followup")
    event(instance, "session.info", {"running": False, "pending": []})
    assert instance.state.busy
    assert [
        params["text"] for method, params, _ in instance.backend.requests if method == "prompt.submit"
    ] == ["first"]
    event(instance, "message.complete", {"text": "first reply"}, admission_id="a")
    assert instance.state.queued == ["local followup"]
    event(
        instance,
        "session.info",
        {
            "running": True,
            "pending": [{"admission_id": "b", "input_id": "peer", "text": "peer", "status": "started"}],
        },
    )
    event(instance, "message.start", {}, admission_id="b")
    event(instance, "session.info", {"running": False, "pending": []})
    assert [
        params["text"] for method, params, _ in instance.backend.requests if method == "prompt.submit"
    ] == ["first"]
    event(instance, "message.complete", {"text": "peer reply"}, admission_id="b")
    assert [
        params["text"] for method, params, _ in instance.backend.requests if method == "prompt.submit"
    ] == ["first", "local followup"]
    assert [row.text for row in instance.state.rows if row.kind == "assistant"] == [
        "first reply",
        "peer reply",
    ]


def test_resume_hint_round_trips_shell_arguments_and_active_gateway(monkeypatch):
    from hermes_for_tern import launcher

    instance = App(
        SimpleNamespace(caps=SimpleNamespace(cols=100)),
        [],
        Path("/project"),
        {
            "profile": "work's profile",
            "gateway_host": "chosen-host",
            "gateway_hermes": "/remote/bin/hermes custom",
            "gateway_cwd": "/remote/a project",
        },
    )
    instance.state.stored_session_id = "saved id; $(false)"
    instance.state.title = 'Title "quoted" $HOME; `false`'
    instance.state.begin("work")
    commands = instance.resume_hint().splitlines()[1:]
    flags = [
        "-p",
        "work's profile",
        "--gateway-hermes",
        "/remote/bin/hermes custom",
        "--gateway-host",
        "chosen-host",
        "--gateway-cwd",
        "/remote/a project",
    ]
    assert shlex.split(commands[0]) == ["hermes", "--resume", "saved id; $(false)", *flags]
    assert shlex.split(commands[1]) == ["hermes", "-c", 'Title "quoted" $HOME; `false`', *flags]
    monkeypatch.setattr(
        launcher, "settings", lambda: {"original": "/stale/hermes", "gateway": {"host": "stale-host"}}
    )
    instance.doctor()
    gateway = next(row.text for row in instance.state.rows if row.text.startswith("gateway: "))
    assert "chosen-host" in gateway and "/remote/a project" in gateway and "stale-host" not in gateway
    demo = App(SimpleNamespace(), [], Path("/project"), {"demo": 1, "gateway_host": "ignored"})
    demo.state.stored_session_id = "saved"
    demo.state.begin("work")
    assert shlex.split(demo.resume_hint().splitlines()[1]) == ["hermes", "--resume", "saved"]


def test_fence_refresh_preserves_clarification_draft_and_multiselect_until_generation_changes():
    instance = ready_app()
    prompt = {
        "kind": "clarify",
        "prompt_id": "q",
        "execution_generation": 3,
        "question": "Which?",
        "choices": ["A", "B"],
        "multi_select": True,
    }
    event(instance, "clarify.request", prompt)
    instance.choose("q", "A")
    instance.answer_draft = Draft("typed answer", 12)
    instance.switch_model("m")
    answer(instance, "session.resume", snapshot(running=True, prompts=[prompt]))
    assert instance.answer_draft.text == "typed answer"
    event(instance, "clarify.request", prompt)
    assert instance.answer_draft.text == "typed answer"
    instance.choose("q", None)
    assert instance.backend.requests[-1][:2] == (
        "clarify.respond",
        {"session_id": "live", "prompt_id": "q", "answer": "A", "execution_generation": 3},
    )
    event(instance, "clarify.request", {**prompt, "execution_generation": 4})
    assert instance.answer_draft.text == ""
    instance.choose("q", "B")
    instance.choose("q", None)
    assert instance.backend.requests[-1][1]["answer"] == "B"
    older = next(
        future
        for method, params, future in instance.backend.requests
        if method == "clarify.respond" and params["execution_generation"] == 3
    )
    older.set_result({"prompt_id": "q"})
    instance.check_requests()
    assert instance.state.clarify().generation == 4
    event(instance, "clarify.settled", {"prompt_id": "q", "execution_generation": 3})
    assert instance.state.clarify().generation == 4
    assert instance.backend.requests[-1][1]["answer"] == "B"


def test_history_snapshot_discards_only_events_already_in_its_projection():
    instance = resuming_app(resume="live")
    event(instance, "message.delta", {"text": "already projected"}, replay_epoch="epoch", seq=1)
    instance.backend.events.put(
        {
            "method": "event",
            "params": {
                "session_id": "live",
                "type": "message.delta",
                "payload": {"text": " new"},
                "replay_epoch": "epoch",
                "seq": 2,
            },
        }
    )
    answer(
        instance,
        "session.resume",
        snapshot(
            last_sequence=1, running=True, display_messages=[{"role": "assistant", "text": "projected"}]
        ),
    )
    instance.read_backend()
    assert [row.text for row in instance.state.rows if row.kind == "assistant"] == ["projected", " new"]
    assert instance.attachment.last_sequence == 2 and not instance.state.unknown


def test_fence_refresh_during_idle_final_gap_does_not_make_sending_ready():
    instance = ready_app()
    instance.submit("work")
    instance.switch_model("m")
    answer(instance, "session.resume", snapshot(running=False, pending=[]))
    instance.submit("held")
    assert instance.state.busy and instance.state.queued == ["held"]
    assert [method for method, _, _ in instance.backend.requests].count("prompt.submit") == 1
    event(instance, "message.complete", {"text": "finished"})
    assert [
        params["text"] for method, params, _ in instance.backend.requests if method == "prompt.submit"
    ] == ["work", "held"]


def test_cancelled_queued_admission_does_not_finish_the_active_reply():
    instance = ready_app()
    instance.submit("active")
    submission = instance.backend.requests[-1][1]["submission_id"]
    event(
        instance,
        "session.info",
        {
            "running": True,
            "pending": [
                {"admission_id": "a", "input_id": submission, "text": "active", "status": "started"},
                {"admission_id": "cancelled", "input_id": "peer", "text": "peer", "status": "queued"},
            ],
        },
    )
    event(instance, "message.delta", {"text": "still"})
    instance.submit("held")
    event(
        instance,
        "session.info",
        {
            "running": True,
            "pending": [{"admission_id": "a", "input_id": submission, "text": "active", "status": "started"}],
        },
    )
    event(instance, "message.complete", {"text": "", "outcome": "cancelled"}, admission_id="cancelled")
    event(instance, "message.delta", {"text": " writing"})
    assert instance.state.busy and instance.state.queued == ["held"]
    assert [row.text for row in instance.state.rows if row.kind == "assistant"] == ["still writing"]
    assert not any(row.kind == "delivered" for row in instance.state.rows)
    event(instance, "session.info", {"running": False, "pending": []})
    event(instance, "message.complete", {"text": "finished"}, admission_id="a")
    assert [
        params["text"] for method, params, _ in instance.backend.requests if method == "prompt.submit"
    ] == ["active", "held"]


def test_model_command_separates_explicit_provider_from_model_id():
    for raw in ("gpt-6-luna --provider openai-codex", "--provider=openai-codex gpt-6-luna"):
        instance = ready_app()
        instance.run_command("/model " + raw)
        settled(instance, snapshot())
        params = instance.backend.requests[-1][1]
        assert params["payload"] == {"model": "gpt-6-luna", "provider": "openai-codex"}
        settled(
            instance, {"operation": "model", "model": "gpt-6-luna", "provider": "openai-codex", "revision": 8}
        )
        assert instance.state.info["model"] == "gpt-6-luna"
        assert instance.state.info["provider"] == "openai-codex"


def test_catalog_pick_uses_its_provider_even_for_a_unique_model_name():
    instance = ready_app()
    instance.state.models = CATALOG
    instance.switch_model("claude-haiku-4")
    settled(instance, snapshot())
    assert instance.backend.requests[-1][1]["payload"] == {"model": "claude-haiku-4", "provider": "anthropic"}

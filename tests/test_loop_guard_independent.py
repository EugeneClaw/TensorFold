"""Independent CPU regressions for PR #262; no trained model or Metal kernels."""

import json

import mlx.core as mx
import numpy as np
import pytest
from http_fakes import post

from tensorfold.engine.lane_engine import LaneStream
from tensorfold.server.loop_guard import LoopGuard
from tests.lane_fakes import VOCAB, FakeEngine, FakeFamily
from tests.test_loop_guard_app import LOOPY, make_loopy_app

PROMPT = [1, 2, 3]
END, NL, NLNL = 90, 91, 92

def next_token(history, period=2):
    """Repetition while thinking; after closing, depend on the whole cache history."""
    reply = history[len(PROMPT):]
    if END in reply:
        if len(reply) - reply.index(END) > 8:
            return 0
        return 1 + (sum(history) * 7 + len(history) * 3) % 35
    if len(reply) < 70:
        return 1 + (len(reply) * 11) % 35
    return 40 + (len(reply) - 70) % period

class RepeatingFamily(FakeFamily):
    def __init__(self, period=2, gpu=False):
        super().__init__()
        self.period = period
        self.gpu_tokens = gpu

    def hidden(self, inputs, cache, parents=None):
        picks = []
        for token in np.array(inputs).reshape(-1).tolist():
            cache[0].rows[0].append(int(token))
            picks.append(next_token(cache[0].rows[0], self.period))
        return mx.array(picks, dtype=mx.float32).reshape(1, -1, 1)

class CopyProposer:
    last_match = 10000

    def __init__(self, period=2, wrong_every=0):
        self.period, self.wrong_every = period, wrong_every

    def propose(self, context, max_draft):
        history, out = list(context), []
        for index in range(max_draft):
            value = next_token(history, self.period)
            if self.wrong_every and (index + 1) % self.wrong_every == 0:
                value = (value + 1) % VOCAB
            out.append(value)
            history.append(value)
        return out

def run_cycle(*, drafts=False, period=2, wrong_every=0, gpu=False, guard=True,
              max_new=600, budget=0, constraint=None):
    engine = FakeEngine(RepeatingFamily(period, gpu), retain_finished_caches=True,
                        max_rows=16, max_draft=15)
    stream = LaneStream(
        stream_id="probe", prompt_ids=PROMPT.copy(), max_new_tokens=max_new,
        eos_ids=frozenset({0}), think_close=(NL, END, NLNL), think_end=END,
        think_open=True, think_budget=budget, loop_guard=LoopGuard() if guard else None,
        drafts=drafts, proposer=CopyProposer(period, wrong_every) if drafts else None,
        constraint=constraint,
    )
    engine.add_stream(stream)
    while engine.active_count:
        engine.step()
    return stream, engine

@pytest.mark.parametrize("streaming", [False, True])
def test_http_guard_does_not_exceed_max_tokens_on_fire(streaming):
    app = make_loopy_app(loop_guard=True)
    try:
        status, body = post(app, {"model": "fake-loopy", "messages": LOOPY,
                                  "max_tokens": 322, "stream": streaming,
                                  "stream_options": {"include_usage": True}})
        assert status == 200
        if streaming:
            chunks = [json.loads(line[6:]) for line in body.splitlines()\
                      if line.startswith("data: ") and line != "data: [DONE]"]
            usage = next(chunk["usage"] for chunk in chunks if chunk.get("usage"))
        else:
            usage = json.loads(body)["usage"]
        assert usage["completion_tokens"] <= 322, usage
    finally:
        app.close()

def test_period_two_continuation_matches_on_both_http_response_modes(monkeypatch):
    import tests.test_loop_guard_app as fixture

    monkeypatch.setattr(fixture, "CYCLE", (88, 89))
    replies = []
    for streaming in [False, True]:
        app = make_loopy_app(loop_guard=True)
        try:
            status, body = post(app, {"model": "fake-loopy", "messages": LOOPY,
                                      "max_tokens": 900, "stream": streaming,
                                      "stream_options": {"include_usage": True}})
            assert status == 200
            if streaming:
                assert "data: [DONE]" in body
                chunks = [json.loads(line[6:]) for line in body.splitlines()\
                          if line.startswith("data: ") and line != "data: [DONE]"]
                assert not any("error" in chunk for chunk in chunks)
                final = next(chunk for chunk in chunks if chunk.get("tensorfold"))
                content = "".join(choice.get("delta", {}).get("content") or ""
                                  for chunk in chunks for choice in chunk.get("choices", []))
                reasoning = "".join(choice.get("delta", {}).get("reasoning_content") or ""
                                  for chunk in chunks for choice in chunk.get("choices", []))
                reason = next(choice["finish_reason"] for chunk in chunks
                              for choice in chunk.get("choices", []) if choice.get("finish_reason"))
            else:
                final = json.loads(body)
                message = final["choices"][0]["message"]
                content, reasoning = message["content"], message.get("reasoning_content", "")
                reason = final["choices"][0]["finish_reason"]
            assert content and reasoning and reason == "stop"
            assert final["tensorfold"]["loop"] == {"period": 2}
            replies.append((content, reasoning, final["usage"], final["tensorfold"]["token_sha"]))
        finally:
            app.close()
    assert replies[0] == replies[1]

@pytest.mark.parametrize("period", [1, 2, 8])
@pytest.mark.parametrize("wrong_every", [0, 3])
@pytest.mark.parametrize("gpu", [False, True])
def test_drafted_answer_matches_serial_after_loop_close(period, wrong_every, gpu):
    serial, _ = run_cycle(period=period, gpu=gpu)
    drafted, engine = run_cycle(drafts=True, period=period, wrong_every=wrong_every, gpu=gpu)
    assert serial.loop == drafted.loop == {"period": period}
    assert drafted.accepted > 0
    assert serial.finish_reason == drafted.finish_reason == "stop"
    assert drafted.emitted == serial.emitted
    tokens, cache = engine.finished_caches["probe"]
    assert cache[0].rows[0] == tokens

@pytest.mark.parametrize("period", [1, 2, 8])
@pytest.mark.parametrize("gpu", [False, True])
@pytest.mark.parametrize("budget", [0, 100])
def test_without_guard_serial_and_drafted_controls_match(period, gpu, budget):
    serial, _ = run_cycle(period=period, gpu=gpu, guard=False, budget=budget)
    drafted, engine = run_cycle(drafts=True, period=period, gpu=gpu,
                                guard=False, budget=budget)
    assert serial.emitted == drafted.emitted
    assert serial.finish_reason == drafted.finish_reason
    assert serial.loop is drafted.loop is None
    tokens, cache = engine.finished_caches["probe"]
    assert cache[0].rows[0] == tokens

@pytest.mark.parametrize("drafts", [False, True])
def test_cache_after_guard_close_can_resume_a_followup(drafts):
    _, engine = run_cycle(drafts=drafts)
    tokens, cache = engine.finished_caches["probe"]
    followup = LaneStream(stream_id="followup", prompt_ids=[*tokens, 80], max_new_tokens=10,
                          eos_ids=frozenset({0}), drafts=False)
    resumed = FakeEngine(RepeatingFamily())
    resumed.add_stream(followup, cache=cache, cached_tokens=len(tokens))
    while resumed.active_count:
        resumed.step()
    cold = LaneStream(stream_id="cold", prompt_ids=[*tokens, 80], max_new_tokens=10,
                      eos_ids=frozenset({0}), drafts=False)
    fresh = FakeEngine(RepeatingFamily())
    fresh.add_stream(cold)
    while fresh.active_count:
        fresh.step()
    assert followup.emitted == cold.emitted

@pytest.mark.parametrize("drafts", [False, True])
def test_guard_close_allows_a_valid_structured_answer(drafts):
    import xgrammar as xgr

    from tensorfold.engine.grammar import Grammars, Spec

    vocab = ["<eos>"] + [chr(0x4E00 + i) for i in range(1, VOCAB)]
    vocab[END], vocab[NL], vocab[NLNL] = "</think>", "\n", "\n\n"
    vocab[1], vocab[2] = "O", "K"
    grammars = Grammars(xgr.TokenizerInfo(vocab, xgr.VocabType.RAW,
                                       vocab_size=VOCAB, stop_token_ids=[0]))
    compiled = grammars.compile(Spec("choice", '["OK"]'))
    # Positive control: the existing budget close hands off to the actual grammar.
    control, _ = run_cycle(drafts=drafts, guard=False, budget=100,
                           constraint=grammars.constraint(compiled, think_end=END))
    assert control.finish_reason == "stop" and control.constraint.finished
    assert control.emitted[control.emitted.index(END) + 1:] == [1, 2, 0]
    stream, _ = run_cycle(drafts=drafts,
                          constraint=grammars.constraint(compiled, think_end=END))
    assert stream.loop == {"period": 2}
    assert stream.finish_reason == "stop", str(stream.error)
    assert stream.constraint.finished
    assert stream.emitted[stream.emitted.index(END) + 1:] == [1, 2, 0]

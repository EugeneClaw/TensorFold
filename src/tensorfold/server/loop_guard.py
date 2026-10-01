"""Opt-in guard that ends a reply stuck repeating one short token cycle inside its think block.

Issue #204: a reasoning turn repeated one token for ~28,000 tokens until the request's
cap. The guard watches committed reply tokens while the think block is open, and fires
when the tail becomes an exact cycle of period <= 8 that has repeated for
``LOOP_MIN_RUN`` tokens (the 256th consecutive predecessor equality). It never runs on
visible content, never decodes on the hot path (text confirmation only on a candidate),
and is a pure function of the emitted prefix, so a greedy replay re-fires at the same
token. Off by default; the server wires one per app and passes it per request.
"""

from __future__ import annotations

import contextlib
from typing import Any

MAX_PERIOD = 8             # longest token cycle considered (a period-8 cycle = 8-token block)
LOOP_MIN_RUN = 256         # consecutive predecessor equalities required to fire
WARM_IN = 64               # the cycle must start at least this far into the reply


class LoopGuard:
    """Token-space cycle detector; ``check`` returns (period, run) or None."""

    def __init__(self, tokenizer: Any = None, lock: Any = None,
                 max_period: int = MAX_PERIOD, min_run: int = LOOP_MIN_RUN, warm_in: int = WARM_IN) -> None:
        self.tokenizer, self.lock = tokenizer, lock
        self.max_period, self.min_run, self.warm_in = int(max_period), int(min_run), int(warm_in)

    def check(self, emitted: list[int]) -> tuple[int, int] | None:
        """Fire test over the committed reply tail. Cheap int compares; decode only on a candidate."""

        n = len(emitted)
        if n < self.min_run + 1:
            return None
        for period in range(1, self.max_period + 1):
            if n < self.min_run + period:
                break
            # the detected cycle region is the last min_run + period tokens; warm-in requires
            # it to start at least warm_in tokens into the reply (only tightens as period grows)
            if n - self.min_run - period < self.warm_in:
                break
            # the 256th consecutive predecessor equality: emitted[-k] == emitted[-k-period] for k = 1..min_run
            run = 0
            for k in range(1, self.min_run + 1):
                if emitted[-k] != emitted[-k - period]:
                    break
                run = k
            if run == self.min_run and self._text_confirmed(emitted, period):
                return period, self.min_run
        return None

    def _text_confirmed(self, emitted: list[int], period: int) -> bool:
        """Decode-sanity gate on the matched region. Periodicity itself is structural: each
        token's bytes are fixed, so a token-periodic run decodes to byte-periodic, hence
        text-periodic, output — a token-cycle that decodes aperiodically is not constructible.
        What CAN happen is a decoder anomaly (byte-shard garbage, replacement-char runs), so
        the gate rejects regions whose decode is empty or dominated by U+FFFD. The first
        period is deliberately not used as a shift anchor: BPE boundary effects there would
        create false negatives on genuine loops (review round 1, finding 5).
        """

        if self.tokenizer is None:
            return True
        region = emitted[-(self.min_run + period):]
        with contextlib.nullcontext(self.lock):
            text = self.tokenizer.decode(region)
        bad = text.count("\ufffd")
        return bool(text) and bad <= len(text) // 2

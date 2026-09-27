"""SpecPrefill: attention-scored sparse prefill for GLM-5.3-Flash on TensorFold.

Env switch: TF_SPEC=1 enables; TF_SPEC_KEEP (0.4), TF_SPEC_THRESHOLD (8192),
TF_SPEC_FLATNESS (1.0 = gate off; below 1.0, skip spec when chunk-importance
entropy >= the value — no-signal content falls back to the exact dense feed),
TF_SPEC_DRAFT (4 lookahead steps), TF_SPEC_LOG (1) tune it. The whole-prompt
activation gate (length > threshold) is decided in SerialEngine._feed.
"""

from .spec_feed import make_feed
from .spec_scorer import (
    flatness,
    signal_stats,
    spec_enabled as enabled,
    spec_contrast_gate,
    spec_flatness_gate,
    spec_keep_pct,
    spec_log,
    spec_threshold,
)

__all__ = [
    "make_feed",
    "enabled",
    "flatness",
    "signal_stats",
    "spec_contrast_gate",
    "spec_flatness_gate",
    "spec_keep_pct",
    "spec_log",
    "spec_threshold",
]

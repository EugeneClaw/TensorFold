"""SpecPrefill: attention-scored sparse prefill for GLM-5.3-Flash on TensorFold.

Env switch: TF_SPEC=1 enables; TF_SPEC_KEEP (0.4), TF_SPEC_THRESHOLD (8192),
TF_SPEC_DRAFT (4 lookahead steps), TF_SPEC_LOG (1) tune it. The whole-prompt
activation gate (length > threshold) is decided in SerialEngine._feed.
"""

from .spec_feed import make_feed
from .spec_scorer import spec_enabled as enabled, spec_keep_pct, spec_threshold

__all__ = ["make_feed", "enabled", "spec_keep_pct", "spec_threshold"]

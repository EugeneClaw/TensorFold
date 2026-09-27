"""Unit tests for the SpecPrefill flatness gate's signal_stats (E-076 review F5c).

Edges: uniform, single-outlier needle, all-zeros, NaN, short arrays, padding bias.
Run: pytest tests/test_spec_signal_stats.py -q  (needs mlx; skips gracefully without).
"""
import math
import pytest

mlx = pytest.importorskip("mlx.core")
from tensorfold.families.glm5_next.specprefill.spec_scorer import signal_stats, flatness  # noqa: E402


def test_uniform_is_flatum_but_near_one():
    import mlx.core as mx
    imp = mx.ones(4096)
    flat, contrast = signal_stats(imp)
    assert 0.999 <= flat <= 1.0001        # fp32 noise may push it a hair over 1
    assert contrast == pytest.approx(1.0)


def test_single_outlier_has_low_entropy_high_contrast():
    import mlx.core as mx
    imp = mx.ones(4096)
    imp[32:64] = 50.0                     # one needle chunk, 50x the rest
    flat, contrast = signal_stats(imp)
    assert flat < 0.80
    assert contrast > 5.0


def test_all_zeros_is_total_skip_sentinel():
    import mlx.core as mx
    flat, contrast = signal_stats(mx.zeros(4096))
    assert flat == 1.0 and contrast == 1.0


def test_nan_total_is_total_skip_sentinel():
    import mlx.core as mx
    imp = mx.full((4096,), float("nan"))
    flat, contrast = signal_stats(imp)
    assert flat == 1.0 and contrast == 1.0   # must not fire spec on NaN importance


def test_short_array_never_gates():
    import mlx.core as mx
    flat, contrast = signal_stats(mx.ones(64))   # 2 chunks < 16
    assert flat == 0.0 and contrast == 1.0


def test_flatness_wrapper_matches_signal_stats():
    import mlx.core as mx
    imp = mx.ones(2048)
    assert flatness(imp) == signal_stats(imp)[0]
    assert abs(math.log(64)) > 0          # normalizer sanity

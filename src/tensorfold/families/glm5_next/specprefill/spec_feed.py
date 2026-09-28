"""Sparse prompt prefill for GLM-5.3-Flash: feed kept tokens, advance caches by the full length.

The dual of the scorer: only the selected rows run through the layers, while
every cache advances by the FULL prompt length (positions are implicit —
cache.offset IS the position book):

- MLA layers: every prompt row is appended exactly once, in order — real
  latents for kept rows, zero latents with a very-negative indexer gate for
  skipped rows. The indexer pool cannot select gap blocks (finite -1e4
  sentinel: -9984 after the bf16 cast, ~7936 nats below any real score; its
  pool weight exp(-9984 - top) is exactly 0.0 in fp32), and the prefill sdpa
  mask EXCLUDES gap keys in every window (bool mask): zero latents give gap
  keys logit exactly 0 (E-079 measured: quantized_matmul of a zero latent
  returns exactly 0.0 — no additive bias), so an unmasked gap key would
  contribute exp(0)=1 to the softmax denominator and nothing to the value
  mixture — pure mass dilution, ~n_gap/(n_gap + Σexp(l)) attenuation.
- KDA layers: exact by construction — the recurrence runs per kept SEGMENT
  (the fused kernel, fed only real rows), and each maximal gap run of length
  L advances the state in closed form with the run-length power of the null
  gate g_null = exp(lb·sigmoid(A·dt_bias)) (a zero-content row's gate), with
  the conv window zeroed across runs (a gap token contributes zeros).

The generation seed is the feed's OWN last row, final-normed (what
``model.hidden`` would return for it): the last prompt token is never fed
twice. The MTP head's cache absorbs exactly the fed rows that HAVE a known
successor, with their true next-tokens (oMLX's sparse absorb).
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from tensorfold.families.glm5_next.model import KDA, KDA_K, MLACache, hc_expand, project

from .spec_scorer import (SpecScorer, select_chunks, signal_stats,
                          spec_keep_pct, spec_flatness_gate, spec_contrast_gate,
                          spec_anchor_gate, spec_log)

GAP_IG = -1.0e4          # bf16 -> -9984 exactly; finite so pool_blocks never NaNs


def align_keep(keep: np.ndarray, kpool: int = 4) -> np.ndarray:
    """Kept mask -> feed mask: feed the last row of every gap run of >= kpool rows, plus the final row."""
    assert len(keep) >= kpool, "align_keep needs at least kpool rows"
    keep = keep.copy()
    keep[-1] = True
    gap = ~keep
    # gap-run-end markers via shifted ANDs (vectorized; E-079: was a kpool-iteration Python loop)
    run = np.ones(len(keep), dtype=bool)
    for s in range(1, kpool):
        run &= np.concatenate([np.zeros(s, dtype=bool), gap[:-s]])
    run_end = np.zeros_like(keep)
    run_end[:-1] = run[:-1] & keep[1:]      # gap-run of >= kpool ending at j, kept row at j+1: feed j
    return keep | run_end


def pad_caches(cache: list) -> None:
    """Give MLACaches empty real arrays so copies/evals see them (born zero-padded)."""
    for c in cache:
        if isinstance(c, MLACache) and c.keys is None:
            c.keys = mx.zeros((0, 512), dtype=mx.bfloat16)
            c.ik = mx.zeros((0, 128), dtype=mx.bfloat16)
            c.ig = mx.zeros((0, 128), dtype=mx.bfloat16)
            c.pool = mx.zeros((0, 128), dtype=mx.bfloat16)


def _scatter_rows(real: mx.array, rel: mx.array, span: int, fill=None) -> mx.array:
    """Span-sized array: zeros (or scalar ``fill``) everywhere, ``real`` rows at ``rel``."""
    if fill is None:
        out = mx.zeros((span,) + tuple(real.shape[1:]), dtype=real.dtype)
    else:
        out = mx.full((span,) + tuple(real.shape[1:]), fill, dtype=real.dtype)
    out[rel] = real
    return out


def _g_null(kda: KDA) -> mx.array:
    """A zero-content row's gate [H, D]: exp(lb · sigmoid(A·dt_bias))."""
    return mx.exp(kda.cfg.linear_lower_bound
                  * mx.sigmoid(kda.A * kda.dt_bias)).astype(mx.float32)


def _kda_branch(kda: KDA, c, z0: mx.array, rows_abs: np.ndarray, span: int, w0: int) -> mx.array:
    """One KDA sub-layer, exact over the full span.

    The fused kernel runs per maximal run of CONSECUTIVE absolute fed positions; every
    absolute gap of length L between runs advances the state in closed form with the
    null-gate power (g_null^L — a zero-content row's exact attenuation), and the conv
    window is zeroed across gaps (a gap token contributes zeros). ``kda_rows`` applies
    f_b/g_b/o_norm/gating internally — its y is ready for o_proj.
    """
    width3 = 3 * kda.width
    conv = c.conv if c.conv is not None else mx.zeros((kda.taps - 1, width3), dtype=mx.bfloat16)
    entry = c.ssm if c.ssm is not None else mx.zeros((1, kda.heads, kda.dim, kda.dim),
                                                     dtype=mx.float32)
    gn = _g_null(kda)
    outs: list[mx.array] = []

    def run_seg(lo: int, hi: int) -> None:
        nonlocal conv, entry
        if hi > lo:
            proj_seg = project(z0[lo:hi], kda.in_proj, rows_exact=False)
            y, entry, conv = KDA_K.kda_rows(kda, proj_seg, conv, entry)
            outs.append(project(y, kda.o_proj, rows_exact=False))

    def decay(L: int) -> None:
        nonlocal conv, entry
        if L > 0:
            entry = entry * mx.power(gn, float(L)).reshape(1, kda.heads, 1, kda.dim)
            conv = mx.zeros((kda.taps - 1, width3), dtype=mx.bfloat16)

    decay(int(rows_abs[0]) - w0)                          # leading gap (from the previous window)
    seg = 0
    # segment boundaries vectorized (E-079: was a per-row Python loop — 360K numpy-scalar
    # iterations at 8K fed rows; np.flatnonzero on the diff is one pass)
    breaks = np.flatnonzero(np.diff(rows_abs) != 1)
    for b in breaks.tolist():
        run_seg(seg, b + 1)
        decay(int(rows_abs[b + 1]) - int(rows_abs[b]) - 1)
        seg = b + 1
    run_seg(seg, len(rows_abs))
    c.ssm, c.conv = entry, conv
    c.offset += span
    c._replay = None
    return mx.concatenate(outs) if len(outs) > 1 else outs[0]


def _mla_branch(attn, c: MLACache, z0: mx.array, rel: mx.array, span: int,
                prior_fed: np.ndarray | None = None) -> mx.array:
    """One sparse-MLA sub-layer: full-span stamped append + kept-rows attention.

    The sdpa mask excludes gap keys explicitly (bool mask, keep semantics) — in EVERY
    window: ``prior_fed`` carries the absolute positions of earlier windows' fed rows
    (E-079: prior-window gap keys were accidentally left valid before; they are exact
    zeros — zero latent through the quantized wk/wv yields logit 0 and value 0 — so the
    effect was pure softmax-mass dilution, an untracked ~(1-frac) attenuation that
    varied by window depth. Masked now, as the module docstring always claimed).
    """
    cfg = attn.cfg
    rows = int(z0.shape[0])
    parts = [project(z0, p, rows_exact=False) for p in (attn.q_a, attn.kv_a, attn.ik_proj, attn.iw)]
    qr = mx.fast.rms_norm(parts[0], attn.q_norm, cfg.rms_norm_eps)
    q = project(qr, attn.q_b, rows_exact=False).reshape(rows, attn.heads, attn.nope)
    lat = mx.fast.rms_norm(parts[1], attn.kv_norm, cfg.rms_norm_eps)
    ik = mx.fast.layer_norm(parts[2], attn.ik_norm_w, attn.ik_norm_b, 1e-6)
    ig = z0.astype(mx.float32) @ attn.igate.astype(mx.float32)
    lat = _scatter_rows(lat.astype(mx.bfloat16), rel, span)
    ik = _scatter_rows(ik.astype(mx.bfloat16), rel, span)
    ig = _scatter_rows(ig.astype(mx.bfloat16), rel, span, mx.array(GAP_IG, mx.bfloat16))
    c.append(lat, ik, ig, attn.ape, cfg.index_kpool)          # advances offset by the full span
    end = c.offset
    start = end - span
    k, v = attn.keys_values(c.keys[:end])                     # [H, end, dk]
    q_pos = start + rel
    causal = mx.arange(end)[None] <= q_pos[:, None]
    valid = mx.zeros((end,), dtype=mx.bool_)
    if prior_fed is not None and len(prior_fed):
        valid[mx.array(prior_fed.astype(np.int64))] = True    # earlier windows: fed rows only (true now)
    valid[start + rel] = True
    mask = causal & valid[None, :]
    o = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k[None], v[None],
                                             scale=attn.scale, mask=mask[None, None])
    out = o[0].transpose(1, 0, 2).reshape(rows, -1)
    return project(out, attn.o_proj, rows_exact=False)


def feed_sparse(model, tokens: list[int], keep_idx, cache: list, step: int = 2048):
    """Feed kept tokens; every cache advances by the FULL prompt length.
    Returns (keep mask, feed mask, collapsed raw hidden per fed row)."""
    n = len(tokens)
    keep = np.zeros(n, dtype=bool)
    keep[np.asarray(keep_idx, dtype=np.int64)] = True
    feed_mask = align_keep(keep)
    idx = np.nonzero(feed_mask)[0]
    pad_caches(cache)
    raws: list = []
    prior_fed: np.ndarray | None = None                       # absolute fed positions of windows < w0
    for w0 in range(0, n, step):
        w1 = min(w0 + step, n)
        rows = idx[(idx >= w0) & (idx < w1)]                  # fed rows of this ABSOLUTE window
        span = w1 - w0
        rel = mx.array((rows - w0).astype(np.int64))
        chunk = [int(tokens[j]) for j in rows]
        x = model.embed_tokens(mx.array(chunk).reshape(-1).astype(mx.uint32))
        x = mx.contiguous(mx.broadcast_to(x[:, None, :], (len(chunk), model.args.hc_mult, x.shape[-1])))
        for i, (layer, c) in enumerate(zip(model.layers, cache)):
            xc, post, comb = layer.attn_hc.split(x, False)
            z0 = mx.fast.rms_norm(xc, layer.in_norm, layer.eps)
            if isinstance(layer.attn, KDA):
                out = _kda_branch(layer.attn, c, z0, rows, span, w0)
            else:
                out = _mla_branch(layer.attn, c, z0, rel, span, prior_fed)
            x = hc_expand(out, x, post, comb, False)
            xc, post, comb = layer.ffn_hc.split(x, False)
            fb = layer.mlp(mx.fast.rms_norm(xc, layer.post_norm, layer.eps), False)
            x = hc_expand(fb, x, post, comb, False)
            if i % 8 == 7:
                mx.async_eval(x)
        raws_chunk = x.astype(mx.float32)
        raw = raws_chunk[:, 0]
        for s in range(1, int(x.shape[1])):
            raw = raw + raws_chunk[:, s]
        raws.append((raw * (1.0 / int(x.shape[1]))).astype(x.dtype))
        prior_fed = idx[idx < w1]                             # for the next window's mask
        arrays = [x]
        for c in cache:
            arrays.extend(a for a in c.state if a is not None)
        mx.eval(*arrays)
    return keep, feed_mask, (mx.concatenate(raws) if len(raws) > 1 else raws[0])


def make_feed(glmflash):
    """Whole-prompt spec feed: ``feed(tokens, cache) -> hidden`` (the engine calls this for
    prompts above the threshold; scoring, sparse feed, MTP absorb and the seed all happen here)."""
    scorer = SpecScorer(glmflash)

    class Feed:
        def feed(self, tokens, cache):
            started = time.perf_counter()
            n = len(tokens)
            tokens = list(tokens)
            importance, score_s, scorer_cache = scorer.score_tokens(tokens, return_cache=True)
            flat, contrast = signal_stats(importance)
            gate = spec_flatness_gate()
            cgate = spec_contrast_gate()
            flat_hit = flat >= gate
            contrast_hit = contrast < cgate
            anchor = None
            agate = spec_anchor_gate()
            if (flat_hit or contrast_hit) and agate > 0.0:
                # E-078: opt-in anchor bypass (explicit-question traffic only — see
                # spec_anchor_gate). With the gate disabled (default), a flat/contrast
                # hit always punts to the exact dense path (correctness-first).
                anchor = scorer.anchor_strength(tokens, scorer_cache)
                if anchor < agate:
                    if spec_log():
                        arm = "flatness" if flat_hit else "contrast"
                        print(f"[spec] skip ({arm}: flat {flat:.3f} vs {gate:.2f}, "
                              f"contrast {contrast:.2f} vs {cgate:.2f}, anchor {anchor:.2f} "
                              f"vs {agate:.2f}): {n} tokens dense; score {score_s:.2f}s wasted",
                              flush=True)
                    return None
                if spec_log():
                    print(f"[spec] flat-but-anchored (flat {flat:.3f}, anchor {anchor:.2f}): "
                          f"selection runs", flush=True)
            elif flat_hit or contrast_hit:
                if spec_log():
                    arm = "flatness" if flat_hit else "contrast"
                    print(f"[spec] skip ({arm}: flat {flat:.3f} vs {gate:.2f}, "
                          f"contrast {contrast:.2f} vs {cgate:.2f}): {n} tokens dense; "
                          f"score {score_s:.2f}s wasted", flush=True)
                return None
            keep_idx = select_chunks(importance, keep_pct=spec_keep_pct())
            keep, feed_mask, raws = feed_sparse(glmflash.model, tokens, keep_idx, cache, step=2048)
            # the MTP cache sees exactly the fed rows that HAVE a known successor (F6: the last
            # fed row's next token is generated, not known)
            glmflash._raw = raws
            fed_pos = np.nonzero(feed_mask)[0]
            body = fed_pos[:-1]
            if len(body):
                nxt_tokens = np.array([tokens[min(j + 1, n - 1)] for j in body], dtype=np.int64)
                glmflash.absorb_draft_context(raws[:-1], nxt_tokens, cache)
            # final-norm the feed's own last row: exactly what model.hidden returns for the seed
            hidden = mx.fast.rms_norm(raws[-1:], glmflash.model.norm,
                                      glmflash.model.args.rms_norm_eps)[None]
            mx.eval(hidden)
            fed = int(np.count_nonzero(feed_mask))
            if spec_log():
                print(f"[spec] prompt {n}: kept {len(keep_idx)}/{n} "
                      f"({len(keep_idx) / n * 100:.0f}%, fed {fed} rows after alignment) "
                      f"score {score_s:.2f}s, feed {time.perf_counter() - started - score_s:.2f}s",
                      flush=True)
            return hidden

    return Feed()

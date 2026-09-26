"""Sparse prompt prefill for GLM-5.3-Flash: feed kept tokens, advance caches by the full length.

The dual of the scorer: only the selected rows run through the layers, while
every cache advances by the FULL prompt length —

- MLA layers: every prompt row is appended exactly once, in order — real
  latents for kept rows, zero latents with a very-negative indexer gate for
  skipped rows (keys built from zero latents are zero; the indexer never
  selects them). Attention queries are computed for kept rows only, so the
  sub-layer cost stays O(kept x span). The pool lifecycle of
  ``MLACache.append`` is reused as-is: contiguous appends complete
  boundary-straddling blocks on the next chunk.
- KDA layers: skipped rows enter the recurrence with zero q/k/v/g (a pure
  decay step — beta·v·k^T = 0) and their ghost output read is scattered back
  to zero; kept rows run the real kernel on the same zero-gap context.

The final prompt row is always re-fed through ``model.hidden`` (the engine's
own one-row path) so the generation seed is exactly what the serial engine
would hold. Positions are implicit — cache.offset IS the position book — so
the full-length offset advance makes every later mask/index correct.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from tensorfold.families.glm5_next.model import KDA, KDA_K, MLACache, hc_expand, project

from .spec_scorer import SpecScorer, select_chunks, spec_keep_pct, spec_log

GAP_IG = -1.0e4          # finite: a whole -inf 4-pack would NaN pool_blocks' softmax


def align_keep(keep: np.ndarray, kpool: int = 4) -> np.ndarray:
    """Kept mask -> feed mask: feed the last row of every gap run of >= kpool rows, plus the final row."""
    keep = keep.copy()
    keep[-1] = True
    gap = ~keep
    run = np.ones(len(keep), dtype=bool)
    for s in range(kpool):
        run = run & gap if s == 0 else run & np.concatenate([np.zeros(s, dtype=bool), gap[:-s]])
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


def _kda_branch(kda: KDA, c, z0: mx.array, rel: mx.array, span: int) -> mx.array:
    """One KDA sub-layer over a fed chunk, advanced across the full span (gap rows decay only).
    ``kda_rows`` applies f_b/g_b/o_norm/gating internally — its y is ready for o_proj."""
    proj = _scatter_rows(project(z0, kda.in_proj, rows_exact=False), rel, span)
    conv_prev = c.conv if c.conv is not None else mx.zeros((kda.taps - 1, 3 * kda.width), dtype=mx.bfloat16)
    entry = c.ssm if c.ssm is not None else mx.zeros((1, kda.heads, kda.dim, kda.dim), dtype=mx.float32)
    y, ssm, conv_new = KDA_K.kda_rows(kda, proj, conv_prev, entry)
    out = project(y[rel], kda.o_proj, rows_exact=False)       # fed rows only; the state moved `span`
    c.ssm, c.conv, c.offset, c._replay = ssm, mx.contiguous(conv_new), c.offset + span, None
    return out


def _mla_branch(attn, c: MLACache, z0: mx.array, rel: mx.array, span: int) -> mx.array:
    """One sparse-MLA sub-layer: full-span stamped append + kept-rows attention over the sparse cache."""
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
    q_pos = start + rel                                       # absolute positions of the kept rows
    causal = mx.arange(end)[None] <= q_pos[:, None]
    o = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k[None], v[None],
                                             scale=attn.scale, mask=causal[None, None])
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
    abs_pos = idx
    for w0 in range(0, n, step):
        w1 = min(w0 + step, n)
        rows = abs_pos[(abs_pos >= w0) & (abs_pos < w1)]      # fed rows of this ABSOLUTE window
        span = w1 - w0                                        # the window advances the offset by its full length
        begin_abs = w0
        rel = mx.array((rows - begin_abs).astype(np.int64))
        chunk = [int(tokens[j]) for j in rows]
        x = model.embed_tokens(mx.array(chunk).reshape(-1).astype(mx.uint32))
        x = mx.contiguous(mx.broadcast_to(x[:, None, :], (len(chunk), model.args.hc_mult, x.shape[-1])))
        for i, (layer, c) in enumerate(zip(model.layers, cache)):
            xc, post, comb = layer.attn_hc.split(x, False)
            z0 = mx.fast.rms_norm(xc, layer.in_norm, layer.eps)
            if isinstance(layer.attn, KDA):
                ab = _kda_branch(layer.attn, c, z0, rel, span)
            else:
                ab = _mla_branch(layer.attn, c, z0, rel, span)
            x = hc_expand(ab, x, post, comb, False)
            xc, post, comb = layer.ffn_hc.split(x, False)
            z1 = mx.fast.rms_norm(xc, layer.post_norm, layer.eps)
            fb = layer.mlp(z1, False)
            x = hc_expand(fb, x, post, comb, False)
            if i % 8 == 7:
                mx.async_eval(x)
        raws_chunk = x.astype(mx.float32)
        raw = raws_chunk[:, 0]
        for s in range(1, int(x.shape[1])):
            raw = raw + raws_chunk[:, s]
        raws.append((raw * (1.0 / int(x.shape[1]))).astype(x.dtype))
        arrays = [x]
        for c in cache:
            arrays.extend(a for a in c.state if a is not None)
        mx.eval(*arrays)
    return keep, feed_mask, (mx.concatenate(raws) if len(raws) > 1 else raws[0])


def make_feed(glmflash):
    """Whole-prompt spec feed: ``feed(tokens, cache) -> hidden`` (the engine calls this for
    prompts above the threshold; scoring, sparse feed, MTP absorb and the exact seed row
    all happen here)."""
    scorer = SpecScorer(glmflash)

    class Feed:
        def feed(self, tokens, cache):
            started = time.perf_counter()
            n = len(tokens)
            tokens = list(tokens)
            importance, score_s = scorer.score_tokens(tokens)
            keep_idx = select_chunks(importance, keep_pct=spec_keep_pct())
            keep, feed_mask, raws = feed_sparse(glmflash.model, tokens, keep_idx, cache, step=2048)
            # the generation seed is the feed's OWN last row (the final prompt token is fed by
            # feed_sparse exactly once — re-feeding it through model.hidden would duplicate it
            # in every cache and shift all attention); collapse its streams to raw hidden,
            # then the engine's one-row path on the LAST TOKEN OF THE NEXT CHUNK handles decode.
            # The MTP cache sees exactly the rows the backbone actually fed (oMLX's sparse
            # absorb), with their true next-tokens, so drafts keep their prompt context.
            glmflash._raw = raws
            fed_pos = np.nonzero(feed_mask)[0]
            nxt_tokens = np.array([tokens[min(j + 1, n - 1)] for j in fed_pos], dtype=np.int64)
            glmflash.absorb_draft_context(raws, nxt_tokens, cache)
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

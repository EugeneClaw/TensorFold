"""SpecPrefill for the GLM-5.3-Flash family: sparse prompt prefill scored by a one-layer scorer.

Ports the mission's SpecPrefill (oMLX 0.7.0rc1 patches/specprefill.py, E-053
recipe) onto TensorFold's GLM stack. The target prefill feeds only the important
tokens; the caches advance by the FULL prompt length, so every later position
book stays correct (TF positions are implicit: cache.offset is the position).

The scorer is a one-layer LM assembled from tensors the server already holds —
the target's embed/head around the nextn layer (the MTP head's plain layer,
GLMMTP.layer, shared with drafting; no eh_proj, E-025 form). The prompt pass
stores only its latent keys; the attention/MLP tail runs for one seed row and a
short greedy lookahead, capturing queries. Every prompt position is scored by
how hard decoding attends to it:

    importance = mean_lookahead( max_head( avgpool13( softmax(Q K^T) ) ) )

Selection keeps whole 32-token chunks (top keep-pct) plus a mandatory trailing
512-token window, mirroring select_chunks upstream."""

from __future__ import annotations

import math
import os
import time

import mlx.core as mx

from tensorfold.families.glm5_next.model import MLACache, project


def _flag(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def spec_enabled() -> bool:
    return _flag("TF_SPEC", "0") not in ("", "0", "false", "off")


def spec_keep_pct() -> float:
    return float(_flag("TF_SPEC_KEEP", "0.4"))


def spec_threshold() -> int:
    return int(_flag("TF_SPEC_THRESHOLD", "8192"))


def spec_log() -> bool:
    return _flag("TF_SPEC_LOG", "1") not in ("", "0", "false", "off")


def spec_flatness_gate() -> float:
    """Skip spec when chunk-importance entropy exceeds this (scorer has no signal).
    1.0 disables the gate (legacy behavior: always spec above threshold)."""
    return float(_flag("TF_SPEC_FLATNESS", "1.0"))


def spec_contrast_gate() -> float:
    """Skip spec when max/median chunk importance is below this (no outlier = no
    distinctive content for the scorer to lock onto). Default 1.0 = disabled."""
    return float(_flag("TF_SPEC_CONTRAST", "1.0"))


class SpecScorer:
    """One-layer scoring LM: the target's embed/head around the nextn layer's attention (dense causal)."""

    def __init__(self, glmflash) -> None:
        if getattr(glmflash, "mtp", None) is None:
            raise RuntimeError("SpecScorer needs the MTP head's plain layer (drafts enabled)")
        base = glmflash.model                              # the GLM5 backbone
        self.model = base
        self.layer = glmflash.mtp.layer                    # load_layer(nextn, plain=True), stateless
        self.attn = self.layer.attn
        self.eps = base.args.rms_norm_eps
        self.norm = base.norm
        self.lm_head = base.lm_head
        self.lookahead = max(1, int(_flag("TF_SPEC_DRAFT", "4")))
        self.pool_kernel = 13
        self.scale = int(self.attn.cfg.qk_nope_head_dim) ** -0.5

    def _latents(self, tokens, cache: MLACache) -> None:
        """Prompt pass: latent keys only (no attention/MLP tail — the cache is all scoring needs)."""
        attn = self.layer.attn
        ids = mx.array(tokens).reshape(-1).astype(mx.uint32)
        x = self.model.embed_tokens(ids)
        h = mx.fast.rms_norm(x, self.layer.in_norm, self.eps)
        lat = mx.fast.rms_norm(project(h, attn.kv_a, rows_exact=False), attn.kv_norm, self.eps)
        zero = mx.zeros((int(lat.shape[0]), int(attn.ik_norm_w.shape[0])), mx.bfloat16)
        cache.append(lat, zero, zero, attn.ape, attn.cfg.index_kpool)

    def _row(self, token: int, cache: MLACache, append: bool):
        """One scoring row: (out [1, D], q [1, H, nope]); ``append`` stores its latent first."""
        attn = self.layer.attn
        x = self.model.embed_tokens(mx.array([int(token)]).astype(mx.uint32))
        h = mx.fast.rms_norm(x, self.layer.in_norm, self.eps)
        if append:
            lat = mx.fast.rms_norm(project(h, attn.kv_a, rows_exact=False), attn.kv_norm, self.eps)
            zero = mx.zeros((1, int(attn.ik_norm_w.shape[0])), mx.bfloat16)
            cache.append(lat, zero, zero, attn.ape, attn.cfg.index_kpool)
        end = cache.offset                                 # the row sits at end - 1 (causal, self included)
        qr = mx.fast.rms_norm(project(h, attn.q_a, rows_exact=False), attn.q_norm, self.eps)
        q = project(qr, attn.q_b, rows_exact=False).reshape(1, attn.heads, attn.nope)
        k, v = attn.keys_values(cache.keys[:end])          # [H, end, dk]
        mask = mx.arange(end)[None] <= (end - 1)           # bool keep-mask (F4: never an additive float)
        o = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None], k[None], v[None],
                                                 scale=self.scale, mask=mask[None, None])
        o = o[0].transpose(1, 0, 2).reshape(1, -1)
        x = x + project(o, attn.o_proj, rows_exact=False)
        x = x + self.layer.mlp(mx.fast.rms_norm(x, self.layer.post_norm, self.eps), False)
        return x, q

    def _logits(self, rows: mx.array) -> mx.array:
        normed = mx.fast.rms_norm(rows, self.norm, self.eps)
        return project(normed, self.lm_head, rows_exact=True)

    def score_tokens(self, tokens: list[int]):
        """Per-token importance [n] + scoring seconds: oMLX aggregation, greedy lookahead."""
        started = time.perf_counter()
        cache = MLACache()
        n = len(tokens)
        for begin in range(0, n, 2048):
            self._latents(tokens[begin:begin + 2048], cache)
        mx.eval(*cache.state)
        out, q0 = self._row(tokens[-1], cache, append=False)
        token = int(mx.argmax(self._logits(out).reshape(-1)).item())
        qs = [q0]
        for _ in range(self.lookahead):
            out, q = self._row(token, cache, append=True)
            qs.append(q)
            token = int(mx.argmax(self._logits(out).reshape(-1)).item())
            mx.eval(out, *cache.state)
        k, _ = self.attn.keys_values(cache.keys[:n])       # score against PROMPT keys only
        q_stack = mx.concatenate([q.transpose(1, 0, 2) for q in qs], axis=1)   # [H, L, dk]
        scores = (q_stack @ k.transpose(0, 2, 1)) * self.scale                 # [H, L, n]
        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
        if self.pool_kernel > 1:
            weights = _avg_pool1d(weights, self.pool_kernel)
        importance = mx.mean(mx.max(weights, axis=0), axis=0)  # max over heads, then mean over lookahead (oMLX)
        mx.eval(importance)
        return importance, time.perf_counter() - started


def _avg_pool1d(x: mx.array, kernel_size: int) -> mx.array:
    pad = kernel_size // 2
    padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    zeros = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    prefix = mx.concatenate([zeros, mx.cumsum(padded, axis=-1)], axis=-1)
    return (prefix[..., kernel_size:] - prefix[..., :-kernel_size]) / kernel_size


def select_chunks(importance: mx.array, keep_pct: float = 0.4, chunk_size: int = 32,
                  tail_tokens: int = 512) -> mx.array:
    """Top keep-pct chunks by mean importance + mandatory trailing window (drawn from the budget first)."""
    M = int(importance.shape[0])
    if keep_pct >= 1.0:
        return mx.arange(M)
    n_chunks = math.ceil(M / chunk_size)
    keep_n = max(1, math.ceil(n_chunks * keep_pct))
    n_tail = min(n_chunks, math.ceil(tail_tokens / chunk_size)) if tail_tokens > 0 else 0
    ranked_end = n_chunks - n_tail
    if ranked_end > 0:
        pad = ranked_end * chunk_size - M
        imp = importance[: ranked_end * chunk_size] if pad <= 0 else mx.pad(importance, [(0, pad)])
        means = mx.mean(imp.reshape(ranked_end, chunk_size), axis=1).astype(mx.float32).tolist()
    else:
        means = []
    budget = max(0, keep_n - n_tail)
    top = sorted(range(ranked_end), key=lambda i: means[i], reverse=True)[:budget]
    top.extend(range(ranked_end, n_chunks))
    top.sort()
    indices: list[int] = []
    for ci in top:
        indices.extend(range(ci * chunk_size, min(ci * chunk_size + chunk_size, M)))
    return mx.array(indices)


def flatness(importance: mx.array, chunk_size: int = 32) -> float:
    """Normalized entropy of chunk-mean importance in [0, 1].

    ~1.0 = distribution near-uniform (scorer has NO signal: selection is near-random
    and retrieval-critical chunks get dropped with probability ~keep_pct).
    Low values = peaked distribution (scorer found distinctive content).
    Measured on the word-salad worst case: >=0.99. On narrative/needle content: <=0.97.
    """
    return signal_stats(importance, chunk_size)[0]


def signal_stats(importance: mx.array, chunk_size: int = 32) -> tuple[float, float]:
    """(normalized entropy, contrast=max/median chunk importance).

    Entropy barely moves when ONE chunk out of hundreds carries a needle; the
    distribution's tail (max/median) is the outlier detector.
    """
    M = int(importance.shape[0])
    n_chunks = math.ceil(M / chunk_size)
    if n_chunks < 16:
        return 0.0, 1.0
    pad = n_chunks * chunk_size - M
    imp = importance[: n_chunks * chunk_size] if pad <= 0 else mx.pad(importance, [(0, pad)])
    means = mx.mean(imp.reshape(n_chunks, chunk_size), axis=1).astype(mx.float32)
    total = mx.sum(means).item()
    if total <= 0.0:
        return 1.0, 1.0
    p = means / total
    ent = -mx.sum(p * mx.log(p + 1e-12)).item()
    s = mx.sort(means)
    med = max(float(s[n_chunks // 2].item()), 1e-12)
    contrast = float(s[-1].item()) / med
    return ent / math.log(n_chunks), contrast

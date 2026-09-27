# SpecPrefill for GLM-5.3-Flash (glm5_next) — Evidence & Receipts

Environment: Apple M5 Ultra 256 GB, macOS 27.0, venv with mlx 0.32.2, TensorFold PR #9 base (commit 0834afd), checkpoint Vontra/GLM-5.3-Flash-MLX-4bit-MTP. All measurements from a resident serving process, temperature 0 unless noted, unique nonces per request (no cache replay), server-reported token counts. Raw JSON + logs are in this branch, one directory up (`e075/`), harness scripts in `harness/`.

## 1. Cold prefill throughput (tokens/s, n=3 medians)

| prompt tokens | stock TF (PR #9) | + SpecPrefill k=0.4 | + SpecPrefill k=0.3 | oMLX SpecPrefill (reference) |
|---|---|---|---|---|
| ~1.6K | 1163 | 1160 | 1160 | 1071 |
| ~6.3K | 1143 | 1140 | 1146 | 626 |
| ~14K | 1074 | 1871 | 2272 | 2585 |
| ~25K | 867 | 1694 | 2044 | 2584 |
| ~31K | 767 | 1624 | 1983 | — |

Sources: `e075/e072/tf_serial.json` (stock), `e075/e075_prefill_serial.json` (k=0.4), `e075/e075_k03_prefill_serial.json` (k=0.3), `e075/e072/omlx.json` (oMLX). Below the 8,192-token threshold spec never fires — the ~1.6K/6.3K rows are the parity proof.

## 2. Decode throughput (chars/s client-side; ≈3.9 chars/token on this corpus)

| arm | median | min | max | n |
|---|---|---|---|---|
| short prompts | 346.3 (≈ 88.2 tok/s) | 345.2 | 349.3 | 10 |
| mid ~12.9K (spec active) | 382.6 (≈ 80.6 tok/s) | 382.4 | 382.9 | 10 |

Matches the PR #9 decode record within noise. Source: `e075/e075_decode.json`.

## 3. Quality — zero-loss A/B (six task families, deterministic scorers, 2 reps each)

| arm | aggregate | recall | constraints | summary | toolcall | codeedit | arith |
|---|---|---|---|---|---|---|---|
| TF + spec k=0.3 | 0.833 | 1.0 | 0.5 | 1.0 | 0.5 | 1.0 | 1.0 |
| TF no-spec control | 0.833 | 1.0 | 0.5 | 1.0 | 0.5 | 1.0 | 1.0 |
| oMLX + spec (reference) | 0.750 | 1.0 | 0.0 | 1.0 | 0.5 | 1.0 | 1.0 |
| TF + spec k=0.3, flatness gate on | 0.833 | 1.0 | 0.5 | 1.0 | 0.5 | 1.0 | 1.0 |

Spec vs control: identical per-task pattern; failure outputs byte-identical across arms (baseline model/scorer behavior, not spec). Sources: `e075/e075_quality_tf_k03.json`, `e075/e075_quality_tf_ctrl.json`, `e075/e075_quality_omlx.json`, `e075/e075_quality_gated.json`.

## 4. Retrieval (needle) — and the honest failure mode we found

Distinctive-content ladder (verbatim code string embedded at 25/50/75% depth in 6K/12K/24K/30K contexts, 512-token budget): **12/12 PASS** with spec live (`e075/e075_ladder.json`, log `e075_ladder2.log`).

Flat-content soak (60 mixed requests through a hosted server): 0 errors, no throughput decay over the hour, short-context recall 30/30 — but 15/30 long-prompt word-salad needles missed at k=0.3. Fixpoint replay at 512-token budget ruled out response truncation: 14/15 still miss. Dose-response isolated the cause — keep=1.0 passes 5/5 on the same configs (the sparse feed itself is exact); the scorer has no signal when chunk importance is near-uniform, so selection is near-random and a needle chunk survives only with p ≈ keep:

| keep | pass rate on flat-content needles |
|---|---|
| 0.3 | ~50% (15/30 soak; 14/15 confirmed real by replay) |
| 0.5 | 13/20 = 65% |
| 0.7 | 3/5 (small n) |
| 1.0 | 5/5 |

Mitigation shipped in this PR: `TF_SPEC_FLATNESS` signal gate — when normalized chunk-importance entropy ≥ the threshold (default 1.0 = gate off; we serve 0.98), spec skips itself and the request takes the exact dense path. Gated replay of all 15 misses: **15/15 PASS** (server log: `e075/tfserve_gate_skips.log`, flatness 0.985–0.989 on skip decisions). A contrast detector (max/median chunk importance) was tested and rejected — max-over-heads inflates tails on flat content too (12–30× on pure filler).

## 5. Numerics review

The implementation passed an independent high-effort numerics review before the benchmarks above; all findings (gap-key softmax dilution, KDA gap-run conv leak, MTP-cache trim edge, aggregation semantics, last-row absorb) were fixed and the probe suite re-run. Review text: `e075/review_result.txt`. The flatness-gate commit post-dates that review pass.

## 6. Known issues, quarantined rather than hidden

- **Prefix-snapshot aliasing (upstream TF, pre-existing):** snapshot blocks are content-addressed; repetitive prompts alias (we reproduced a resume of 21,179 tokens from a store that never held a 21K sequence). Spec-fed blocks additionally carry keep-gap patterns that are not content-portable. This PR quarantines the interaction (`--snapshot-dir none`, spec sessions never checkpoint, `TF_SPEC_FIRED` canary). Fix direction: key blocks by (content, keep-signature).
- The scorer scores a checkpoint segment without its predecessors' context (accepted approximation; this is the main selection-quality ceiling and the target of the next iteration).

## 7. Reproducing

`harness/` contains the exact scripts: `e072_prefill_tf.py` (cold prefill), `e075_quality.py` (six-task suite), `e075_ladder.py` (needle ladder), `e075_soak.py` + `e075_soak_fixpoint.py` (soak + miss replay), `e075_keepsweep.py` / `e075_rate_ab.py` (keep dose-response). All target an OpenAI-compatible endpoint (`BASE`/`SOAK_BASE` env or in-file constant) with the model id of the served checkpoint.

Config served in these runs: `TF_SPEC=1 TF_SPEC_KEEP=0.3 TF_SPEC_THRESHOLD=8192 TF_SPEC_FLATNESS=0.98 TF_SPEC_LOG=1`, server flags `--context 32768 --no-thinking --snapshot-dir none` (drafts enabled; the scorer reuses the MTP layer that drafts load).

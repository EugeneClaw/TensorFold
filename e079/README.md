# E-079: External review (Minimax M3.1 Preview) — verification receipts

Review context: Minimax reviewed PR #1 as-is (unaware of internal E-076/077/078 work).
Full adjudication: PR comment + LEDGER.md E-079. Summary of what was verified vs fixed:

## Fixed (commit 67e8613)
1. **Mask asymmetry (VALID):** prior-window gap keys were left in the prefill softmax
   (`valid[:start] = True`), contradicting the module docstring. MEASURED: gap keys carry
   exact 0.0 logits AND 0.0 values (quantized_matmul of a zero latent = 0.0 — MLX quant
   `biases` are weight zero-points, not output bias), so the effect was pure softmax-mass
   dilution: each unmasked gap key adds exp(0)=1 to the denominator. Fix: `_mla_branch`
   now masks prior gaps via `prior_fed` (absolute fed positions). Docstring rewritten with
   the measured facts.
2. **TF_SPEC_KEEP default 0.4 → 0.3** (aligned with every receipt).
3. **`_kda_branch` segment loop vectorized** (np.flatnonzero(np.diff != 1)); align_keep
   loop cleaned. Equivalence: old vs new align_keep identical on 200 random masks.
4. Docstrings scoped: "exact by construction" = mechanism-level claim; behavioral battery
   covers result-level.

## Verified after the fix (this receipts branch carries the logs)
- exactness 5/5 byte-identical (`e079_exact.log`)
- narrative needle ladder 12/12 PASS (`e079_ladder.log`; two earlier 0/12 runs were
  output-budget truncation at 64/256 tokens — the model's reasoning didn't fit; at 1024
  the same prompts all pass; ladder protocol now records the required budget)
- quality spot 6/6 with spec active above threshold (`e079_quality_spot` output in ledger)

## Adjudicated claims (not fixed; with reasons)
- "prefer contrast gate": tested E-075g, non-discriminating (one-hot inflation 12.9-158 on
  failing prompts). Flatness discriminates 0.981-0.988 vs <=0.97.
- "gates ship inert": stale — HEAD ships FLATNESS=0.98 (fires constantly in prod logs);
  ANCHOR built+negative+reverted (3/20 vs 11/20, opt-in default 0).
- "zero-loss A/B vacuous": partially fair; per-task ptok recorded, tasks crossed threshold;
  binary-suite limitation acknowledged; content-level battery (n=100 salad, 127K long-arm)
  is the real evidence now.
- "warm vs cold trade": stale — E-076 re-enabled signature-keyed snapshots at PR HEAD.
- Tool-schema needle arm + keep-0.2 suite robustness + spec-vs-snapshot warm A/B: queued.

## Probe scripts
- e079_gapkey_probe2.py — zero-latent through quantized kv_a: max-abs 0.0 (exact);
  typical latent: ~4.0. Softmax-dilution bounds tabled for 25K/90K gap scenarios.

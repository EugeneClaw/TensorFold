# E-079: external review (Minimax M3.1) adjudication — mask fix + receipts

## Claim 1: mask asymmetry — VALID divergence, fixed (docstring was lying)
- **Mechanism correction:** gap keys are NOT "bias-only K/V". Measured directly from the
  checkpoint (layer-11 kv_a quantized weights): `quantized_matmul` of a zero latent returns
  **exactly 0.0** — MLX `biases` are quantization zero-points on W, not an additive output
  bias. Gap keys had logit 0 and value 0.
- **The real defect:** `valid[:start] = True` did include prior-window gap keys in every
  query's softmax (docstring said they're excluded). Each added exp(0)=1 to the denominator,
  nothing to the value mixture → pure mass dilution, ~(1 - n_gap/(n_gap+Σexp(l)))
  attenuation, growing with gap count. Fix: `_mla_branch` now takes `prior_fed` (absolute
  fed positions of earlier windows) and masks prior gaps properly. Docstring updated with
  the measured facts (0.0 result, exp(0)=1 dilution, fp32-underflow proof for the indexer
  sentinel).
- **Why the numbers didn't move:** the dilution was uniform, window-depth-dependent
  attenuation of attention outputs — the model absorbed it (like bf16 noise). Re-verified
  post-fix: exactness 5/5 byte-identical, narrative needle ladder 12/12 PASS.
- Reviewer's instinct ("could be bug or undocumented behaviour; docstring is lying") was
  correct in substance. Good catch.

## Claim 2: gates ship inert — PARTIALLY STALE, now resolved by E-078 work
- The review read the PR at 50cc4bd; HEAD (7b53200+) ships `TF_SPEC_FLATNESS=0.98` default
  in the hosted config, and the gate fires constantly in production logs (punts word-salad
  and repetitive tool output to dense — exactly the reviewer's "wild" case).
- Reviewer's "prefer contrast" was tested and rejected in E-075g: contrast (max/median) is
  one-hot inflated by ANY distinctive chunk (12.9–158 on failing prompts) — it does not
  discriminate. Flatness does (0.981–0.988 fail vs ≤0.97 pass), and E-078 added the
  measured-distribution framing the reviewer asked for (n=100 arms).
- TF_SPEC_ANCHOR (E-078 W1b): built, measured 3/20 vs 11/20 dense (NEGATIVE), reverted,
  shipped opt-in-only with default 0. The negative result is documented in the ledger and
  PR-adjacent receipts rather than hidden.

## Claim 3: zero-loss A/B vacuous — FAIR, partially addressed
- Per-task ptok counts: quality-suite tasks are 3-30K; recall/summary/codeedit/arith tasks
  crossed the 8192 threshold (6 tasks = 6 spec-active runs); the byte-identical results are
  therefore NOT vacuous, but the reviewer is right that a binary suite can't detect small
  content loss. That's why E-078 ran the battery that actually stresses content selection:
  n=100 explicit-question salad (97/100), routing arm, long-arm at 127K (with decomposition).
- keep=0.2 robustness: PENDING (needs a bench window; the daemon runs keep 0.3).
- "Byte-identical is a coarse signal" — agreed and it's now stated as such in receipts
  (E-075c records it as behavior-level evidence, not proof of bitwise cache equality).

## Claim 4 (agent prompts): the scorer is question-conditioned, which E-078 measured
- The "one seed + 4 lookahead rows" description is stale (PR HEAD scores via question-
  conditioned rollout; the P2 probe measured needle rank 0-1/536 with the question in
  context). The reviewer's tool-schema worry is still legitimate and maps to our Arm A/D:
  system+tool blocks in the front are exactly what the always-kept head of the selection +
  indexer tail windows protect; the n=100 battery passes on that shape. A dedicated
  needle-in-tool-schema test is queued (W1c remainder).

## Claim 5 (warm vs cold): partially stale
- E-076 (PR HEAD) already re-enabled snapshots WITH signature-keyed provenance — the
  "quarantine traded warm for cold" concern describes 50cc4bd, not HEAD. Spec fires only
  above 8192 tokens; below that (the bulk of agentic turns) snapshots own the warm path.
  The warm-vs-spec A/B number is queued for the bench window.

## Smaller things — all fixed
- TF_SPEC_KEEP default: 0.4 → **0.3** (aligned with every receipt; E-079).
- `_kda_branch` segment loop: vectorized with `np.flatnonzero(np.diff(rows_abs) != 1)`
  (align_keep loop also cleaned; equivalence proven on 200 random masks).
- "Exact by construction" docstring: reworded to scope the claim to the mechanism (null-gap
  model), pointing at the behavioral battery for result-level equivalence.

## Verification battery after the mask fix (this branch)
- exactness 5/5 byte-identical (e071 harness, sub-threshold + spec arms)
- narrative needle ladder 12/12 PASS (6K-30K × 3 depths, keep 0.3)
- align_keep old≡new on 200 random masks
- quality spot: 6/6 with spec active on the >8192 arm

## Not yet done (queued)
- keep=0.2 suite robustness run (bench window)
- needle-in-tool-schema dedicated arm
- spec-on-warm vs snapshot-on-warm A/B (bench window)

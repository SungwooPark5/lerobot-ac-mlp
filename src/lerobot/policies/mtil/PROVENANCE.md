# MTIL baseline — provenance & faithful-vs-adapted

**Reference**: MTIL — "Encoding Full History with Mamba for Temporal Imitation
Learning", Yulin Zhou et al., arXiv:2505.12410 (IEEE RA-L 2025).
Official code: https://github.com/yulinzhouZYL/MTIL (verified 2026-06-11, `test/`:
`mamba_policy_par.py`, `inference_par.py`, `evaluate_model_transfer.py`).

This directory is an **external baseline reproduced inside our lerobot framework** so
it trains/evaluates on the **same dataset, sim, tasks, and metrics** as our ACM2
carry-spectrum models — the only difference is the temporal model. Running it inside
our harness (rather than MTIL's separate repo on a different sim/data pipeline) is the
fairer, apples-to-apples comparison a top-venue reviewer expects.

## Faithful to MTIL (verified against their code)
| MTIL property | Where in their code | Reproduced here |
|---|---|---|
| **Mamba-2** SSM (not Mamba-3) | `MambaConfig` d_model=2048/d_state=512, `mamba_chunk_scan_combined` | `mamba_ssm.Mamba2` stack |
| Recurrent **history encoder** over the obs stream | `init_hidden_states` + `step()` (1 token/step) | `MTILModel` Mamba-2 stack over the obs-token sequence |
| Predicts a **K-step action chunk** | `future_steps=16`, `out_proj` | `head: Linear(D, K·action_dim)` |
| **No correction/gating/fusion** of state w/ obs | pure `selective_state_update` recurrence | no gate/fusion — plain Mamba-2 scan |
| **Query every step** + **temporal aggregation** | eval `query_frequency=1`, `temporal_agg=True` | `n_action_steps=1` + `_TemporalEnsembler` |
| State reset only **between episodes** | `reset_hiddens()` | `reset()` clears the window |

## Adapted for our framework (honest deviations — state these in the paper)
1. **Vision**: same ResNet-18 backbone as our ACM2 models, NOT MTIL's DINOv2 features.
   Rationale: isolates the *temporal model* as the only difference (fair). DINOv2 would
   confound vision quality with temporal modeling.
2. **Capacity (param-match)**: MTIL has no Transformer encoder / CVAE, so at the
   ACM2 token width (512, 4 layers) it is only ~19M vs our ~48M models. To remove
   the "baseline is smaller" confound we scale its Mamba stack (dim_model=768,
   n_mamba_layers=9) to **46.53M** (measured) — within ~3.4% of the m2 cluster
   (~48M) and the closest integer-layer match. Report this true count in the
   param table; the exact capacity control is the internal m2_* spectrum, not MTIL.
   ⚠️ 2026-06-12 backbone switch Mamba-3→Mamba-2: re-run `count_params_flops.py` on the
   cluster to refresh the exact m2_* counts (Mamba-2 per-layer params differ from Mamba-3).
3. **History length**: trained on a **bounded observation window** of `n_obs_steps`
   frames (default 16). MTIL trains on full-length episode sequences. Set `n_obs_steps`
   large to shrink the gap; our memory tasks (RememberColor cue→use gap ~10) fit within 16.
3. **Deterministic head** (no CVAE) — matches MTIL (it has no CVAE).
4. **Train/inference**: inference uses **unbounded recurrent state carry** across the
   whole episode (`unbounded_carry=True`, via `Mamba2` + `InferenceParams` stepping) —
   faithful to MTIL's defining O(1)-memory unbounded history. Training uses a bounded
   obs-history window (`n_obs_steps=32`); the residual train(window)/infer(unbounded)
   gap is the one honest deviation — set n_obs_steps ≥ the longest train-time dependency
   to shrink it. A bounded-window inference fallback (`unbounded_carry=False`) is kept
   for robustness. (Fully faithful = full-episode per-position supervised training — a
   heavier data-pipeline change; documented as the faithful-max follow-up.)

## Hyperparameters (verified against their code, 2026-06-12)
- **LR = 1e-4, weight_decay = 1e-4, AdamW** — matches MTIL `train/train_par.py` (the
  parallel/`*_par` variant we reproduce; their non-parallel `train.py` uses 2e-4/5e-4).
- ⚠️ Difference: MTIL uses a **CosineAnnealingLR** schedule; ours currently has no
  scheduler (`get_scheduler_preset → None`). Add cosine for closer faithfulness, or
  cover via the LR sweep. Minor.

## NOT code-identical (important — state honestly)
This is a faithful reproduction of MTIL's *method*, NOT a copy of their repository.
Differences: vision (our ResNet18 vs their DINOv2 features), framework (lerobot vs
standalone), training (bounded window vs full-trajectory sequences), size (~46.5M vs
their d_model=2048), scheduler (none vs cosine). Same: Mamba-2 SSM, history-encoder,
per-step query + temporal aggregation, unbounded carry, no correction, LR 1e-4.
Code-identical would mean their sim/data/eval pipeline → an unfair cross-pipeline
comparison; the same-pipeline port is the fair baseline. If a code-identical number is
wanted, cite MTIL's *reported* numbers separately (cannot share our results table).

## How to cite in the paper
> "We compare against MTIL [cite], a Mamba-2 recurrent history-encoder policy that
> carries an uncorrected SSM state. For a controlled comparison we reproduce MTIL's
> temporal model in our framework (same ResNet backbone, dataset, sim, and metrics);
> see App. X for faithful-vs-adapted details."

Distinguish from `m2_lit`: `m2_lit` is the *uncorrected-carry ablation within our ACM2
chunk-decoder* (Mamba-2, open-loop chunk execution); `mtil` is the *faithful MTIL
architecture* (Mamba-2, per-step history encoder + temporal aggregation). Both lack
state correction — the property our method adds.

## ✅ Verified against primary sources (2026-06-12 deep-research, 20 sources, 3-vote adversarial)

Every MTIL setup fact below was confirmed by **direct quotes** from the MTIL paper
(arXiv:2505.12410 v1), the official repo (github.com/yulinzhouZYL/MTIL), the original
ACT paper (arXiv:2304.13705), the act repo (github.com/tonyzhaozh/act), and lerobot
dataset cards. Use these verbatim for the paper / rebuttal — do NOT paraphrase from memory.

### MTIL reported numbers (ACT-sim benchmark, Table I) — CONFIRMED 3-0
| Method | Cube Transfer | Bimanual Insertion |
|---|---|---|
| ACT | 90.0 ± 2.0 | 50.0 ± 3.5 |
| MTIL (10-step history) | 92.0 ± 1.5 | 56.0 ± 2.5 |
| **MTIL (Full history)** | **100.0 ± 0.0** | **84.0 ± 2.1** |

3 seeds, single RTX 4090. **The "100%" is ONE cell: Cube Transfer, Full-history.**
Insertion is 84% even in their best config; with 10-step history MTIL ≈ ACT (92/56 vs 90/50).

### MTIL's self-strengthened setup (= why the headline is high) — CONFIRMED
- **Demos**: **100 *scripted-policy* demonstrations/task**, ~400 steps (Appendix A.1).
  ⚠️ scripted, NOT human — cleaner/more consistent than our 50 *human* demos.
- **Vision**: headline rows use **frozen DINOv2 ViT-L/14** (1024-dim, patch 14);
  ResNet18 is reported only as a "fair comparison with ACT" ablation (Fig 2b, App A.1).
  (Table I has no backbone column → the 100%-row==DINOv2 link is a well-grounded
  inference from App A.1 + Fig 2b, not a printed cell.)
- **Capacity**: d_model=2048, d_state=512, 4 Mamba layers, K=50 (paper Table 6).
  Code/paper gap: code default `future_steps=16` (real action chunk); `MambaConfig.chunk_size=256`
  is the SSM scan chunk, not the action chunk — "chunk_size" is overloaded in their code.
- **Optimizer**: AdamW + CosineAnnealingLR. train.py LR=2e-4/wd=5e-4; train_par.py LR=1e-4/wd=1e-4.
- **Eval**: closed-loop per-step (query_frequency=1, env.step every t over 400 steps) + temporal ensemble.
- **Training budget (epochs)**: paper states **50 epochs** *for LIBERO only* ("results
  averaged over 3 seeds (100, 200, 300) at 50 epochs"; verified ar5iv 2026-06-22). The
  ACT-sim (Cube Transfer / Insertion) epoch count is **not stated** in the paper.
  ⚠️ Do NOT frame "few epochs" as a weakness in the paper/rebuttal: (1) 50 ep is the
  LIBERO benchmark's standard protocol (used for fair baseline comparison), (2) epoch is a
  different unit from our gradient steps (`STEPS=150_000`) — not directly comparable, and
  (3) action-chunk dense supervision + frozen DINOv2 + scripted demos make BC converge in
  few epochs. **Unit conversion** (per-frame windowed sampling, both pipelines):
  `steps ≈ epochs × (n_demos × ep_len) / batch`. On OUR data (50 demos × ~400 frames,
  batch 8): 1 epoch ≈ 2,500 steps → 50 epochs ≈ **~125k steps ≈ our 150k** (same order;
  ours slightly larger). Caveat: the 50-ep figure is LIBERO data (not ALOHA), so this is an
  order-of-magnitude equivalence, not an exact match. The real fairness axes are
  vision/demos/capacity, NOT the epoch number.

### Standard ACT/ALOHA practice (= what WE use) — CONFIRMED 3-0
- **50 demonstrations/task** (both scripted and human variants), ResNet18 backbone,
  **chunk_size k=100** — original ACT paper (arXiv:2304.13705) + act repo constants.py
  (`num_episodes=50`) + lerobot `aloha_sim_transfer_cube_human` info.json (`total_episodes=50`).

### Verdict (Q12) — CONFIRMED 3-0, with one reverse nuance
Our **50 human demos + ResNet18 + chunk100 + param-match + same-pipeline** IS the genuine
ACT/ALOHA standard. MTIL's **100 scripted + DINOv2 + d2048 + full-history** is strengthened
above it → cite their 100%/84% separately, never in our results table.
- 🔴 **Honest reverse (do NOT overclaim)**: MTIL ALSO ran a **ResNet18 param-matched ablation
  that still beat ACT**. So MTIL's gain is NOT solely from the strengthened config — `mtil`
  is a *real* competitor, not a strawman. Our contribution is the **correction** (m2_cor vs
  m2_lit) and **efficiency** (1/K), NOT beating mtil on clean SR.
- ⛔ **Unverified — do NOT claim**: that Robomimic's 1.00 cells are "saturated / Diffusion
  Policy also 1.00" (could not confirm from a primary quote; left open).

### Paper-ready framing sentence
> "The ACT/ALOHA standard is 50 demos, ResNet18, chunk 100 [Zhao 2023]; we reproduce all
> baselines in this standard pipeline with param-matching. MTIL's reported 100.0/84.0
> [MTIL 2025, Table I] uses 100 scripted demos, a frozen DINOv2 ViT-L/14, d_model=2048, and
> full history; we cite it separately and reproduce MTIL's *temporal model* under the standard
> setup (matching their own ResNet18 fair-comparison ablation)."

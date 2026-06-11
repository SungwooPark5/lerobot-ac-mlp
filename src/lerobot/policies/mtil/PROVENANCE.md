# MTIL baseline — provenance & faithful-vs-adapted

**Reference**: MTIL — "Encoding Full History with Mamba for Temporal Imitation
Learning", Yulin Zhou et al., arXiv:2505.12410 (IEEE RA-L 2025).
Official code: https://github.com/yulinzhouZYL/MTIL (verified 2026-06-11, `test/`:
`mamba_policy_par.py`, `inference_par.py`, `evaluate_model_transfer.py`).

This directory is an **external baseline reproduced inside our lerobot framework** so
it trains/evaluates on the **same dataset, sim, tasks, and metrics** as our ACM3
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
1. **Vision**: same ResNet-18 backbone as our ACM3 models, NOT MTIL's DINOv2 features.
   Rationale: isolates the *temporal model* as the only difference (fair). DINOv2 would
   confound vision quality with temporal modeling.
2. **Capacity (param-match)**: MTIL has no Transformer encoder / CVAE, so at the
   ACM3 token width (512, 4 layers) it is only ~19M vs our ~48M models. To remove
   the "baseline is smaller" confound we scale its Mamba stack (dim_model=768,
   n_mamba_layers=9) to **46.53M** (measured) — within ~3.4% of the m3 cluster
   (47.98–48.19M) and the closest integer-layer match. Report this true count in the
   param table; the exact capacity control is the internal m3_* spectrum, not MTIL.
3. **History length**: trained on a **bounded observation window** of `n_obs_steps`
   frames (default 16). MTIL trains on full-length episode sequences. Set `n_obs_steps`
   large to shrink the gap; our memory tasks (RememberColor cue→use gap ~10) fit within 16.
3. **Deterministic head** (no CVAE) — matches MTIL (it has no CVAE).
4. **Train/inference**: both use the bounded-window Mamba scan over the obs tokens
   (consistent). MTIL's unbounded per-step state carry can be added later via
   `Mamba2`'s `InferenceParams.step`; not needed for the targeted memory horizons.

## How to cite in the paper
> "We compare against MTIL [cite], a Mamba-2 recurrent history-encoder policy that
> carries an uncorrected SSM state. For a controlled comparison we reproduce MTIL's
> temporal model in our framework (same ResNet backbone, dataset, sim, and metrics);
> see App. X for faithful-vs-adapted details."

Distinguish from `m3_lit`: `m3_lit` is the *uncorrected-carry ablation within our ACM3
chunk-decoder* (Mamba-3, open-loop chunk execution); `mtil` is the *faithful MTIL
architecture* (Mamba-2, per-step history encoder + temporal aggregation). Both lack
state correction — the property our method adds.

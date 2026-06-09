# Research Overview — Action Chunking Mamba-3

> **Research Question**  
> Transformer-based ACT treats the action decoder as a global attention mechanism with no notion of sequence order or persistent state.  
> Mamba processes tokens sequentially and maintains a hidden state across positions.  
> Do these structural differences create exploitable inductive biases for robot action chunking?

> **Claim**: Explicitly exploiting Mamba's sequential processing (C1: ICPE) and persistent hidden state (C2: SSCP) yields measurable improvements in robot policy performance — and these effects are **SSM-specific**, not achievable by applying the same techniques to a Transformer.

---

## 1. Policy Hierarchy

```
PreTrainedConfig
├── ACTConfig                                    "act"
│   └── ACTICPEConfig                            "act_icpe"                   ← N1 control
├── ACMConfig                                    "acm"
├── ACM2Config                                   "acm2"
└── ACM3Config                                   "acm3"
    ├── ACM3BiMambaConfig                        "acm3_bimamba"               ← E1
    ├── ACM3SelfAttenConfig                      "acm3_self_atten"            ← E2
    ├── ACM3SSCPConfig                           "acm3_sscp"
    └── ACM3ICPEConfig                           "acm3_icpe"
        └── ACM3ICPESSCPConfig                   "acm3_icpe_sscp"             ← ⭐ proposed
            ├── ACM3ICPESSCPBiMambaConfig        "acm3_icpe_sscp_bimamba"     ← E1+C ⭐
            └── ACM3ICPESSCPSelfAttenConfig      "acm3_icpe_sscp_self_atten"  ← E2+C
```

---

## 2. Architecture

### ACM3 Backbone

```
Observation  ──► ResNet18 ──────►┐
Robot State  ──► Linear   ──────►├──► Transformer Encoder (4L, d=512) ──► encoder_out (T, B, D)
VAE Latent   ──► Linear   ──────►┘                                               │
                                                                                  ▼
Action Queries (zeros) ──► cat([encoder_out | queries]) ──► Mamba3 Decoder ──► Action Head ──► â (B, K, A)
```

- **Encoder**: 4-layer Transformer, d=512, 8 heads — same as ACT
- **Decoder**: Mamba-3 SSM (d_state=128, expand=2, headdim=64, RoPE, no d_conv)
- **VAE**: CVAE, kl_weight=10.0 — active during training only
- **Output**: K-step action chunk in one forward pass

---

### C1 — Intra-Chunk Phase Embedding (ICPE)

```
φ(i, K) = [sin(2πi/K),  cos(2πi/K),  i/K,  (K−i)/K]   ∈ ℝ⁴

decoder_in[i] = pos_embed[i]  +  icpe_proj(φ(i, K))
                 (learnable)       (linear, σ_init = 0.1/√D)
```

Mamba processes position i after accumulating h₀…h_{i−1}. Without positional signal, the model cannot distinguish "early in chunk" from "late in chunk" from its inputs alone — ICPE provides this explicitly.  
**SSM-specificity test**: applying the same signal to ACT (N1) should yield no improvement.

---

### C2 — SSCP + Chunk-Continuation (CC)

```
Inference:
  carry_n  =  decoder_out_n[:, −1:, :]                    # (B, 1, D)
  input_n1 =  cat([carry_n,  encoder_out_n1,  queries])

Training (CC, p = 0.5 per batch):
  carry    =  stop_grad(decoder_out_n[:, −1:, :])
  input_n1 =  cat([carry, encoder_out_n1, queries])
  loss     =  L1(chunk_n)  +  L1(chunk_n+1)
```

carry_n is processed first by Mamba's sequential scan, warm-starting hidden state h before encoder tokens arrive. In a Transformer this token is merely an additional attention key — no hidden-state warm-up.

---

### E1 — BiMamba (Vim-style)

```
combined  =  cat([carry, encoder_out, queries])       # optional carry
fwd       =  ForwardLayers(combined)
bwd       =  flip(BackwardLayers(flip(combined)))
out       =  0.5 × (fwd + bwd)                        # preserves activation scale
```

Two independent Mamba3 stacks, no shared weights. Carry prepended to both directions.

---

### E2 — Post-Mamba Self-Attention

```
attn_out    =  MHA(decoder_out, decoder_out, decoder_out)   # (K, B, D)
decoder_out =  LayerNorm(decoder_out  +  tanh(γ) × attn_out)
γ_init      =  1e-4   →   tanh(γ) ≈ 1e-4 at step 0         # near-zero gate
```

Gate initialised near zero so training starts identical to ACM3 baseline.

---

## 3. Research Hypotheses

| # | Hypothesis | Verified by |
|---|-----------|------------|
| **H1** | ICPE reduces temporal jitter at chunk boundaries (end_mid_ratio ↓) | A1–A3 vs B4 |
| **H2** | ICPE is SSM-specific: same signal on Transformer yields no gain | N1 vs B1 |
| **H3** | SSCP reduces inter-chunk discontinuities (spike_ratio ↓) | A4, A4cc vs B4 |
| **H4** | ICPE + SSCP are complementary; joint gain > sum of parts | P1cc vs A3+A4cc |
| **H5** | O(N) SSM complexity provides scaling advantage at large K | K1–K6 |
| **H6** | BiMamba improves over unidirectional but remains below C-series | E1a vs B4, E1b vs P1cc |
| **H7** | Self-Attn provides marginal benefit, confirming C-series captures the critical signal | E2a vs B4, E2b vs P1cc |

---

## 4. Evaluation Metrics

### Primary
| Metric | Definition | Direction |
|--------|-----------|-----------|
| **SR** | Success Rate — fraction of rollouts reaching goal within episode | ↑ |
| **SR@50** | SR evaluated at 50 rollouts (fast sweep) | ↑ |

### Smoothness
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **spike_ratio** | mean(‖aₜ₊₁ − aₜ‖₂ > τ) / B4 baseline | < 1.0 = smoother than B4 |
| **end_mid_ratio** | MSE(a_{K−1}) / MSE(a_{K/2}) | < 1.0 = less chunk-end jitter |

> τ = 95th percentile of ‖aₜ₊₁ − aₜ‖₂ under B4, fixed per task.

### Efficiency (K-Scaling)
| Metric | Definition |
|--------|-----------|
| **SR@K** | SR at K ∈ {50, 100, 200} |
| **train_time@K** | Wall-clock training time per epoch at each K |

---

## 5. Ablation Design

```
B4  (acm3, unidirectional, no phase, no carry)
 │
 ├─ + ICPE sincos only ──────────────────────────────── A1    ┐
 ├─ + ICPE linear only ──────────────────────────────── A2    ├ isolate ICPE signal components (H1)
 ├─ + ICPE full 4D ──────────────────────────────────── A3    ┘
 │       └── same signal on Transformer ─────────────── N1      SSM-specificity proof (H2)
 │
 ├─ + SSCP inference-only ───────────────────────────── A4    ┐
 ├─ + SSCP + CC ─────────────────────────────────────── A4cc  ┘ isolate CC training effect (H3)
 │
 ├─ + ICPE full + SSCP ───────────────────────────────── P1
 └─ + ICPE full + SSCP + CC ─────────────────────────── P1cc  ⭐ (H4)
      │
      ├─ + BiMamba ─────────────────────────────────── E1b    (H6)
      └─ + Self-Attn ───────────────────────────────── E2b    (H7)

B4 (direct)
 ├─ + BiMamba ────────────────────────────────────────── E1a   (H6 baseline)
 └─ + Self-Attn ──────────────────────────────────────── E2a   (H7 baseline)
```

---

## 6. Experiment Table (22 models)

### B — Baselines

| ID | Policy | K | LR | Role | SR | spike_ratio | end_mid_ratio |
|----|--------|---|----|------|----|-------------|---------------|
| **B1** | `act` | 100 | 1e-5 | Transformer baseline | | | |
| **B2** | `acm` | 100 | 3e-5 | Mamba-1 baseline | | | |
| **B3** | `acm2` | 100 | 3e-5 | Mamba-2 baseline | | | |
| **B4** | `acm3` | 100 | 3e-5 | **Mamba-3 primary baseline** | | | |

### N — Control

| ID | Policy | K | LR | Role | SR | spike_ratio | end_mid_ratio |
|----|--------|---|----|------|----|-------------|---------------|
| **N1** | `act_icpe` | 100 | 1e-5 | ICPE on Transformer — expected ≈ B1 | | | |

### A — Ablations

| ID | Policy | Variant | K | Role | SR | spike_ratio | end_mid_ratio |
|----|--------|---------|---|------|----|-------------|---------------|
| **A1** | `acm3_icpe` | sincos | 100 | ICPE: frequency signal only | | | |
| **A2** | `acm3_icpe` | linear | 100 | ICPE: progression signal only | | | |
| **A3** | `acm3_icpe` | full 4D | 100 | ICPE: full φ(i,K) | | | |
| **A4** | `acm3_sscp` | inference | 100 | SSCP without CC training | | | |
| **A4cc** | `acm3_sscp` | CC p=0.5 | 100 | SSCP + Chunk-Continuation | | | |

### P — Proposed

| ID | Policy | CC | K | Role | SR | spike_ratio | end_mid_ratio |
|----|--------|----|---|------|----|-------------|---------------|
| **P1** | `acm3_icpe_sscp` | ✗ | 100 | C1+C2 inference-only | | | |
| **P1cc** | `acm3_icpe_sscp` | ✓ | 100 | **C1+C2 full — proposed method ⭐** | | | |

### K — Scaling

| ID | Policy | K | Role | SR | train_time |
|----|--------|---|------|----|------------|
| **K1** | `act` | 50 | | | |
| **K2** | `act` | 200 | | | |
| **K3** | `acm3` | 50 | | | |
| **K4** | `acm3` | 200 | | | |
| **K5** | `acm3_icpe_sscp` | 50 | | | |
| **K6** | `acm3_icpe_sscp` | 200 | | | |

### E — Extensions

| ID | Policy | K | vs. | Role | SR | spike_ratio |
|----|--------|---|-----|------|----|-------------|
| **E1a** | `acm3_bimamba` | 100 | B4 | BiMamba alone | | |
| **E1b** | `acm3_icpe_sscp_bimamba` ⭐ | 100 | P1cc | BiMamba on C-series | | |
| **E2a** | `acm3_self_atten` | 100 | B4 | Self-Attn alone | | |
| **E2b** | `acm3_icpe_sscp_self_atten` | 100 | P1cc | Self-Attn on C-series | | |

---

## 7. Configuration Reference

### Shared Defaults (ACM3 base)

| Parameter | Value | Note |
|-----------|-------|------|
| `dim_model` | 512 | |
| `n_encoder_layers` | 4 | Transformer |
| `n_decoder_layers` | 1 | Mamba3 |
| `n_heads` | 8 | |
| `chunk_size` | 100 | K-series varies |
| `mamba3_d_state` | 128 | |
| `mamba3_expand` | 2 | inner dim = 1024 |
| `mamba3_headdim` | 64 | |
| `mamba3_rope_fraction` | 0.5 | |
| `mamba3_is_mimo` | False | MIMO needs A100/H100 |
| `mamba3_chunk_size` | 64 | SSD kernel |
| `use_vae` | True | |
| `latent_dim` | 32 | |
| `kl_weight` | 10.0 | |
| `optimizer_lr` | 3e-5 | ACM variants |
| `optimizer_lr` | 1e-5 | ACT variants |
| `optimizer_lr_backbone` | 1e-5 | ResNet18 |

### Policy-Specific Parameters

| Policy | Extra Parameters | Default |
|--------|-----------------|---------|
| `acm3_icpe` | `icpe_mode`, `icpe_scale_init` | `"full"`, `0.1` |
| `acm3_sscp` | `sscp_p_carry`, `sscp_detach` | `0.5`, `True` |
| `acm3_icpe_sscp` | ICPE + SSCP params | — |
| `acm3_self_atten` | `self_atten_nhead`, `self_atten_gamma_init` | `8`, `1e-4` |
| `acm3_bimamba` | — | (decoder-only change) |
| `acm3_icpe_sscp_bimamba` ⭐ | ICPE + SSCP params | — |
| `acm3_icpe_sscp_self_atten` | ICPE + SSCP + `self_atten_*` params | — |

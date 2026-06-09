# Experimental Results

> Environment: `AlohaTransferCube-v0`  
> Dataset: `lerobot/aloha_sim_transfer_cube_human`  
> Eval: n=500 episodes, batch_size=50  
> All results: mean ± 95% CI

---

## 1. Metrics

### 1.1 Primary

| Metric | Code | Formula | Range | Direction |
|--------|------|---------|-------|-----------|
| **SR** | `pc_success / 100` | fraction of episodes where `info["is_success"]` is True | [0, 1] | ↑ |
| **SR@50** | same, n=50 | fast sweep during training | [0, 1] | ↑ |
| **avg_sum_reward** | `avg_sum_reward` | mean accumulated reward per episode | — | ↑ |

### 1.2 Smoothness

| Metric | Code | Formula | Interpretation |
|--------|------|---------|----------------|
| **spike_ratio** | `spike_ratio` | mean\_jerk(boundary ±2 steps) / mean\_jerk(interior 20–80%) | < 1.0 = smoother than baseline; boundary jerk normalized by interior |
| **end\_mid\_ratio** | `end_mid_ratio` | mean\_jerk(end ≥80% of chunk) / mean\_jerk(mid 20–80%) | < 1.0 = less chunk-end jitter |
| **jerk\_abs** | `jerk_abs` | mean \|Δ³aₜ\| | absolute jerk level ↓ |
| **diff\_abs** | `diff_abs` | mean \|aₜ₊₁ − aₜ\|₁ | mean L1 velocity ↓ |
| **SPARC** | `SPARC` | Spectral Arc Length (Balasubramanian 2012), FFT cutoff 10 Hz | −4 (smooth) → −20 (jerky) ↑ |
| **LDJ** | `LDJ` | log(T³ × mean\_jerk² / amplitude²) | more negative = jerkier ↑ |

> **spike_ratio** and **end_mid_ratio** are normalized against B4 (acm3, K=100) as reference.  
> τ (spike threshold) = 95th percentile of ‖aₜ₊₁ − aₜ‖₂ under B4, fixed per task.

### 1.3 Efficiency

| Metric | Code | Definition |
|--------|------|-----------|
| **train\_time@K** | — | Wall-clock training time per epoch at chunk size K |
| **avg\_inference\_latency\_ms** | `avg_inference_latency_ms` | Mean forward pass time (ms), CUDA-synchronized |

### 1.4 Metric Computation Chain

```
rollout()
  └─ per step: action, reward, success, done, latency

eval_policy()
  ├─ mask post-done steps
  ├─ aggregate: SR, avg_sum_reward, avg_max_reward
  └─ save action_logs/*.pt

collect_trajectories.py  (offline, from saved action_logs)
  ├─ jitter (L2 frame-to-frame diff)
  ├─ per-position jerk profile
  ├─ spike_ratio, end_mid_ratio
  ├─ SPARC, LDJ
  └─ save trajectories.npz

common_v6.py
  └─ load + aggregate all above → final metrics dict
```

---

## 2. Main Results

### 2.1 Baseline Comparison

| ID | Policy | SR ↑ | avg\_sum\_reward ↑ | spike\_ratio ↓ | end\_mid\_ratio ↓ |
|----|--------|------|-----------------|--------------|-----------------|
| **B1** | `act` | | | | |
| **B2** | `acm` | | | | |
| **B3** | `acm2` | | | | |
| **B4** | `acm3` | | | 1.00 (ref) | 1.00 (ref) |
| **N1** | `act_icpe` | | | | |

> **N1 vs B1**: ICPE applied to Transformer — expected SR ≈ B1 (SSM-specificity proof).

---

### 2.2 ICPE Ablation (C1)

| ID | ICPE Mode | SR ↑ | spike\_ratio ↓ | end\_mid\_ratio ↓ | SPARC ↑ |
|----|-----------|------|--------------|-----------------|---------|
| **B4** | none | | 1.00 | 1.00 | |
| **A1** | sincos | | | | |
| **A2** | linear | | | | |
| **A3** | full 4D | | | | |

> Expected: A3 > A2 ≈ A1 > B4 on end\_mid\_ratio (ICPE reduces chunk-end jitter).  
> If N1 (ACT+ICPE) ≈ B1, this confirms ICPE is SSM-specific.

---

### 2.3 SSCP Ablation (C2)

| ID | Variant | SR ↑ | spike\_ratio ↓ | end\_mid\_ratio ↓ | SPARC ↑ |
|----|---------|------|--------------|-----------------|---------|
| **B4** | none | | 1.00 | 1.00 | |
| **A4** | SSCP inference-only | | | | |
| **A4cc** | SSCP + CC (p=0.5) | | | | |

> Expected: A4cc > A4 > B4 on spike\_ratio (CC training teaches carry usage).

---

### 2.4 Proposed Method (C1 + C2)

| ID | Variant | SR ↑ | spike\_ratio ↓ | end\_mid\_ratio ↓ | SPARC ↑ | LDJ ↑ |
|----|---------|------|--------------|-----------------|---------|-------|
| **B4** | baseline | | 1.00 | 1.00 | | |
| **A3** | ICPE only | | | | | |
| **A4cc** | SSCP+CC only | | | | | |
| **P1** | ICPE+SSCP (no CC) | | | | | |
| **P1cc** | **ICPE+SSCP+CC** ⭐ | | | | | |

> Expected: P1cc dominates across all metrics. Joint gain (P1cc) > A3 + A4cc − B4 (synergy hypothesis).

---

### 2.5 K-Scaling (C3)

| ID | Policy | K | SR ↑ | train\_time (s/epoch) | avg\_inference\_latency\_ms |
|----|--------|---|------|----------------------|--------------------------|
| **K1** | `act` | 50 | | | |
| **K2** | `act` | 200 | | | |
| **K3** | `acm3` | 50 | | | |
| **K4** | `acm3` | 200 | | | |
| **K5** | `acm3_icpe_sscp` | 50 | | | |
| **K6** | `acm3_icpe_sscp` | 200 | | | |

> Expected: ACT train\_time scales quadratically with K; ACM3 scales linearly.  
> ACM3+ICPE+SSCP SR improves with K at lower compute cost than ACT.

---

### 2.6 E-Series Extensions

| ID | Policy | SR ↑ | spike\_ratio ↓ | end\_mid\_ratio ↓ | Δ SR vs ref |
|----|--------|------|--------------|-----------------|------------|
| **B4** | `acm3` | | 1.00 | 1.00 | — |
| **E1a** | `acm3_bimamba` | | | | vs B4 |
| **E1b** | `acm3_icpe_sscp_bimamba` ⭐ ⭐ | | | | vs P1cc |
| **E2a** | `acm3_self_atten` | | | | vs B4 |
| **E2b** | `acm3_icpe_sscp_self_atten` | | | | vs P1cc |
| **P1cc** | `acm3_icpe_sscp` ⭐ | | | | — |

> **E1b vs P1cc**: Does BiMamba add on top of C-series?  
> **E2b vs P1cc**: Does Self-Attn add anything? (Expected: negligible → C-series is sufficient)

---

## 3. Cross-Model Summary Table

| ID | Policy | SR ↑ | spike\_ratio ↓ | end\_mid\_ratio ↓ | SPARC ↑ |
|----|--------|------|--------------|-----------------|---------|
| **B1** | `act` | | | | |
| **B4** | `acm3` | | | | |
| **N1** | `act_icpe` | | | | |
| **A3** | `acm3_icpe` (full) | | | | |
| **A4cc** | `acm3_sscp` (CC) | | | | |
| **P1cc** | `acm3_icpe_sscp` (CC) ⭐ | | | | |
| **E1a** | `acm3_bimamba` | | | | |
| **E1b** | `acm3_icpe_sscp_bimamba` ⭐ | | | | |
| **E2b** | `acm3_icpe_sscp_self_atten` | | | | |

---

## 4. Training Curves (to be filled)

| ID | Policy | best SR@50 step | final SR@500 | KLD (final) | L1 loss (final) |
|----|--------|----------------|-------------|-------------|-----------------|
| **B1** | `act` | | | | |
| **B4** | `acm3` | | | | |
| **A3** | `acm3_icpe` | | | | |
| **A4cc** | `acm3_sscp` | | | | |
| **P1cc** | `acm3_icpe_sscp` ⭐ | | | | |
| **E1a** | `acm3_bimamba` | | | | |
| **E1b** | `acm3_icpe_sscp_bimamba` ⭐ | | | | |

---

## 5. Qualitative Analysis (to be filled)

### 5.1 Action Trajectory Visualization

| ID | Chunk transition smooth? | End-jitter visible? | Notes |
|----|--------------------------|---------------------|-------|
| **B4** | | | |
| **A3** | | | |
| **P1cc** | | | |
| **E1a** | | | |

### 5.2 SSCP Carry Ablation

| Condition | Observed behavior |
|-----------|------------------|
| carry = None (no warmup) | |
| carry from prev chunk (detached) | |
| carry from prev chunk (gradient) | |

### 5.3 Failure Mode Analysis

| ID | Common failure mode | Frequency |
|----|--------------------|-----------| 
| **B4** | | |
| **P1cc** | | |

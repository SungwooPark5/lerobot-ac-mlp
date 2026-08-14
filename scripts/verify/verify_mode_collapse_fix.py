"""
Mode collapse 해결 검증: Fixed TE vs Residual Learned Ensemble
- rollout 없이 cache 데이터로 offline 분석
- TE가 bimodal 후보들의 gap에 빠지는지
- Residual learned ensemble이 한 mode 쪽으로 이동하는지 확인
"""

import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths / Config
# ============================================================

CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
WEIGHT_NET_PATH = Path("/home1/eunji24/lerobot_project/outputs/weight_network_residual_lam1/weight_net_final.pt")
OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/mode_collapse_verify_lam1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPREAD_THRESHOLD = 0.2
BASE_GAP_TOL = 0.05
MIN_MODE_SEP = 0.1
MIN_SEP_RATIO = 2.0


# ============================================================
# Stage 2와 동일한 클래스
# ============================================================

class ResidualWeightNetwork(nn.Module):
    def __init__(
        self,
        L,
        state_dim,
        action_dim,
        te_coeff=0.01,
        d_model=128,
        n_heads=4,
        n_layers=2,
    ):
        super().__init__()
        self.L = L

        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.queries = nn.Embedding(L, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, 1)

        # 학습 시작점: delta=0, final_logits=TE prior
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.register_buffer(
            "te_logits",
            -te_coeff * torch.arange(L, dtype=torch.float32),
        )

    def forward(self, state, contribs):
        """
        state:    (B, state_dim)
        contribs: (B, L, action_dim)
        """
        B = state.shape[0]

        state_tok = self.state_proj(state).unsqueeze(1)   # (B, 1, D)
        chunk_toks = self.chunk_proj(contribs)            # (B, L, D)

        memory = torch.cat([state_tok, chunk_toks], dim=1)

        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)

        delta = self.head(x).squeeze(-1)                  # (B, L)
        final_logits = self.te_logits.unsqueeze(0) + delta

        return final_logits, delta


# ============================================================
# Utility
# ============================================================

def safe_mean(x):
    x = np.asarray(x)
    if len(x) == 0:
        return float("nan")
    return float(np.mean(x))


def safe_median(x):
    x = np.asarray(x)
    if len(x) == 0:
        return float("nan")
    return float(np.median(x))


def print_stat(name, arr):
    arr = np.asarray(arr)
    if len(arr) == 0:
        print(f"{name}: empty")
        return

    print(
        f"{name}: "
        f"mean={arr.mean():.6f}, "
        f"median={np.median(arr):.6f}, "
        f"min={arr.min():.6f}, "
        f"max={arr.max():.6f}"
    )


def analyze_bimodal_1d(values):
    """
    1D action 후보 values가 진짜 bimodal한지 KMeans로 판정.
    반환:
      valid, modes, sep, sep_ratio
    """
    values = np.asarray(values).reshape(-1, 1)

    if len(values) < 4:
        return False, None, None, None

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(values)
    labels = km.labels_
    centers = km.cluster_centers_.flatten()

    modes = np.sort(centers)
    sep = float(abs(modes[1] - modes[0]))

    if sep < MIN_MODE_SEP:
        return False, modes, sep, None

    within_stds = []
    flat_values = values.flatten()

    for k in [0, 1]:
        vals = flat_values[labels == k]
        if len(vals) >= 2:
            within_stds.append(vals.std())

    if len(within_stds) == 0:
        return False, modes, sep, None

    within = float(np.mean(within_stds))
    sep_ratio = sep / (within + 1e-6)

    if sep_ratio < MIN_SEP_RATIO:
        return False, modes, sep, sep_ratio

    return True, modes, sep, sep_ratio


# ============================================================
# Load cache
# ============================================================

print("=" * 80)
print("Mode Collapse Verify: Fixed TE vs Residual Learned Ensemble")
print("=" * 80)

print("\nLoading cache...")
cache = np.load(CACHE_PATH)

chunks = cache["chunks"]
states = cache["states"]
ep_starts = cache["ep_starts"]

N, chunk_size, action_dim = chunks.shape
state_dim = states.shape[1]

print(f"  cache path : {CACHE_PATH}")
print(f"  chunks     : {chunks.shape}")
print(f"  states     : {states.shape}")
print(f"  ep_starts  : {ep_starts.shape}")
print(f"  N={N}, chunk_size={chunk_size}, action_dim={action_dim}, state_dim={state_dim}")


# ============================================================
# Load model
# ============================================================

print(f"\nLoading weight network from {WEIGHT_NET_PATH}")
ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)

L = int(ckpt["L"])
ckpt_te_coeff = float(ckpt["te_coeff"])

model = ResidualWeightNetwork(
    L=L,
    state_dim=int(ckpt["state_dim"]),
    action_dim=int(ckpt["action_dim"]),
    te_coeff=ckpt_te_coeff,
    d_model=int(ckpt["d_model"]),
    n_heads=int(ckpt["n_heads"]),
    n_layers=int(ckpt["n_layers"]),
).to(DEVICE)

model.load_state_dict(ckpt["model"])
model.eval()

print(f"  L                  : {L}")
print(f"  ckpt te_coeff       : {ckpt_te_coeff}")
print(f"  lambda_delta_reg    : {ckpt.get('lambda_delta_reg', '?')}")
print(f"  d_model             : {ckpt.get('d_model', '?')}")
print(f"  n_heads             : {ckpt.get('n_heads', '?')}")
print(f"  n_layers            : {ckpt.get('n_layers', '?')}")

if L > chunk_size:
    raise ValueError(f"L={L} is larger than chunk_size={chunk_size}. Cannot build full coverage windows.")


# ============================================================
# TE prior weight
# ============================================================

te_weights_const = np.exp(-ckpt_te_coeff * np.arange(L))
te_weights_const = te_weights_const / te_weights_const.sum()

te_logits_const = -ckpt_te_coeff * np.arange(L)

print("\nIndex meaning check:")
print("  contribs[0]  = chunk from frame τ,     position 0     -> newest/current")
print("  contribs[-1] = chunk from frame τ-L+1, position L-1   -> oldest")
print(f"  TE logit[0]  = {te_logits_const[0]:.6f}")
print(f"  TE logit[-1] = {te_logits_const[-1]:.6f}")
print(f"  TE weight[0] = {te_weights_const[0]:.6f}")
print(f"  TE weight[-1]= {te_weights_const[-1]:.6f}")


# ============================================================
# Main offline analysis
# ============================================================

print("\nAnalyzing all full-coverage τ across episodes...")

results = []
full_weight_rows = []

for ep in range(len(ep_starts)):
    ep_from = int(ep_starts[ep][0])
    ep_to = int(ep_starts[ep][1])

    if ep_to - ep_from < L:
        continue

    for win_start in range(ep_from, ep_to - L + 1):
        tau = win_start + L - 1

        # i=0: τ 시점에서 position 0 예측
        # i=1: τ-1 시점에서 position 1 예측
        # ...
        # i=L-1: τ-L+1 시점에서 position L-1 예측
        contribs_frame_idx = tau - np.arange(L)
        pos_idx = np.arange(L)

        contribs_np = chunks[contribs_frame_idx, pos_idx]   # (L, action_dim)
        state_np = states[tau]                              # (state_dim,)

        # Fixed TE output
        te_output = (te_weights_const[:, None] * contribs_np).sum(axis=0)

        # Residual learned ensemble output
        state_t = torch.from_numpy(state_np).float().to(DEVICE).unsqueeze(0)
        contribs_t = torch.from_numpy(contribs_np).float().to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            final_logits, delta = model(state_t, contribs_t)
            weights_t = F.softmax(final_logits, dim=1)

            final_logits_np = final_logits.squeeze(0).detach().cpu().numpy()
            delta_np = delta.squeeze(0).detach().cpu().numpy()
            weights = weights_t.squeeze(0).detach().cpu().numpy()

        res_output = (weights[:, None] * contribs_np).sum(axis=0)

        # candidate spread
        spread_per_dim = contribs_np.std(axis=0)
        spread_max_dim_idx = int(spread_per_dim.argmax())
        spread_max_dim = float(spread_per_dim.max())
        spread_mean = float(spread_per_dim.mean())

        # logits statistics
        sorted_logits = np.sort(final_logits_np)
        logit_gap = float(sorted_logits[-1] - sorted_logits[-2]) if L >= 2 else 0.0
        logit_std = float(final_logits_np.std())
        logit_max = float(final_logits_np.max())
        logit_min = float(final_logits_np.min())

        # weights statistics
        weight_max = float(weights.max())
        weight_argmax = int(weights.argmax())
        entropy = float(-(weights * np.log(weights + 1e-8)).sum())
        effective_k = float(1.0 / (np.sum(weights ** 2) + 1e-8))

        # residual-vs-TE difference
        delta_norm = float(np.mean(delta_np ** 2))
        kl_res_te = float(
            np.sum(weights * (np.log(weights + 1e-8) - np.log(te_weights_const + 1e-8)))
        )
        output_shift_l2 = float(np.linalg.norm(res_output - te_output))
        output_shift_l1 = float(np.mean(np.abs(res_output - te_output)))

        results.append({
            "episode": ep,
            "tau": tau,
            "spread_max_dim": spread_max_dim,
            "spread_max_dim_idx": spread_max_dim_idx,
            "spread_mean": spread_mean,

            "logit_max": logit_max,
            "logit_min": logit_min,
            "logit_std": logit_std,
            "logit_gap": logit_gap,

            "weight_max": weight_max,
            "weight_argmax": weight_argmax,
            "entropy": entropy,
            "effective_k": effective_k,

            "delta_norm": delta_norm,
            "kl_res_te": kl_res_te,
            "output_shift_l2": output_shift_l2,
            "output_shift_l1": output_shift_l1,

            "te_output": te_output,
            "res_output": res_output,
            "contribs": contribs_np,
            "weights": weights,
            "final_logits": final_logits_np,
            "delta": delta_np,
        })

        # 너무 커지는 것 방지: 20 step마다 full weights/logits 저장
        if tau % 20 == 0:
            row = {
                "episode": ep,
                "tau": tau,
                "spread_max_dim": spread_max_dim,
                "spread_max_dim_idx": spread_max_dim_idx,
                "weight_argmax": weight_argmax,
                "weight_max": weight_max,
                "effective_k": effective_k,
                "entropy": entropy,
                "kl_res_te": kl_res_te,
                "output_shift_l2": output_shift_l2,
            }
            for i in range(L):
                row[f"logit_{i}"] = float(final_logits_np[i])
                row[f"delta_{i}"] = float(delta_np[i])
                row[f"weight_{i}"] = float(weights[i])
                row[f"te_weight_{i}"] = float(te_weights_const[i])
            full_weight_rows.append(row)

print(f"  Analyzed {len(results)} τ")

if len(results) == 0:
    raise RuntimeError("No valid full-coverage τ found. Check cache or L.")


# ============================================================
# Convert scalar results to DataFrame
# ============================================================

scalar_rows = []

for r in results:
    scalar_rows.append({
        "episode": r["episode"],
        "tau": r["tau"],
        "spread_max_dim": r["spread_max_dim"],
        "spread_max_dim_idx": r["spread_max_dim_idx"],
        "spread_mean": r["spread_mean"],

        "logit_max": r["logit_max"],
        "logit_min": r["logit_min"],
        "logit_std": r["logit_std"],
        "logit_gap": r["logit_gap"],

        "weight_max": r["weight_max"],
        "weight_argmax": r["weight_argmax"],
        "entropy": r["entropy"],
        "effective_k": r["effective_k"],

        "delta_norm": r["delta_norm"],
        "kl_res_te": r["kl_res_te"],
        "output_shift_l2": r["output_shift_l2"],
        "output_shift_l1": r["output_shift_l1"],
    })

df = pd.DataFrame(scalar_rows)
df_path = OUT_DIR / "mode_collapse_scalar_metrics.csv"
df.to_csv(df_path, index=False)
print(f"\nSaved scalar metrics: {df_path}")

if len(full_weight_rows) > 0:
    full_df = pd.DataFrame(full_weight_rows)
    full_df_path = OUT_DIR / "mode_collapse_full_logits_weights_sampled.csv"
    full_df.to_csv(full_df_path, index=False)
    print(f"Saved sampled logits/weights: {full_df_path}")


# ============================================================
# Decision vs non-decision statistics
# ============================================================

spreads = df["spread_max_dim"].to_numpy()
decision_mask = spreads > SPREAD_THRESHOLD
n_decision = int(decision_mask.sum())
n_total = len(df)

print("\n" + "=" * 80)
print("Decision point statistics")
print("=" * 80)
print(f"Decision points: spread_max_dim > {SPREAD_THRESHOLD}")
print(f"  {n_decision}/{n_total} = {n_decision / max(n_total, 1) * 100:.2f}%")

print("\n[Delta activation]")
print_stat("  delta_norm all", df["delta_norm"])
print_stat("  delta_norm decision", df.loc[decision_mask, "delta_norm"])
print_stat("  delta_norm non-decision", df.loc[~decision_mask, "delta_norm"])

delta_dec = safe_mean(df.loc[decision_mask, "delta_norm"])
delta_non = safe_mean(df.loc[~decision_mask, "delta_norm"])
delta_ratio = delta_dec / max(delta_non, 1e-9)

print(f"  decision/non-decision ratio = {delta_ratio:.3f}x")
print("  guide: >1.5면 decision point에서 residual delta가 더 강하게 활성화된 신호")

print("\n[Weight behavior]")
for col in ["weight_max", "effective_k", "entropy", "logit_std", "logit_gap", "kl_res_te", "output_shift_l2"]:
    print_stat(f"  {col} all", df[col])
    print_stat(f"  {col} decision", df.loc[decision_mask, col])
    print_stat(f"  {col} non-decision", df.loc[~decision_mask, col])
    print("")

print("[Argmax distribution top 20]")
argmax_counts = df["weight_argmax"].value_counts().sort_values(ascending=False)
for idx, count in argmax_counts.head(20).items():
    print(f"  idx={idx:3d}: {count:6d} ({count / len(df) * 100:.2f}%)")


# ============================================================
# Bimodal decision analysis: TE gap vs Residual gap
# ============================================================

print("\n" + "=" * 80)
print("Bimodal decision analysis")
print("=" * 80)

te_in_gap_count = 0
res_in_gap_count = 0
te_in_mode_count = 0
res_in_mode_count = 0
analyzed_bimodal = 0

bimodal_rows = []

for r in results:
    if r["spread_max_dim"] <= SPREAD_THRESHOLD:
        continue

    dim = int(r["spread_max_dim_idx"])
    contribs_dim = r["contribs"][:, dim]

    valid, modes, sep, sep_ratio = analyze_bimodal_1d(contribs_dim)

    if not valid:
        continue

    te_val = float(r["te_output"][dim])
    res_val = float(r["res_output"][dim])

    mode_low = float(modes[0])
    mode_high = float(modes[1])

    # 고정 tol + mode separation 기반 adaptive tol
    tol = max(BASE_GAP_TOL, 0.1 * abs(mode_high - mode_low))

    gap_low = mode_low + tol
    gap_high = mode_high - tol

    te_in_gap = gap_low < te_val < gap_high
    res_in_gap = gap_low < res_val < gap_high

    te_in_mode = (abs(te_val - mode_low) < tol) or (abs(te_val - mode_high) < tol)
    res_in_mode = (abs(res_val - mode_low) < tol) or (abs(res_val - mode_high) < tol)

    te_in_gap_count += int(te_in_gap)
    res_in_gap_count += int(res_in_gap)
    te_in_mode_count += int(te_in_mode)
    res_in_mode_count += int(res_in_mode)
    analyzed_bimodal += 1

    # 어느 mode에 더 가까운지
    te_nearest_mode = 0 if abs(te_val - mode_low) <= abs(te_val - mode_high) else 1
    res_nearest_mode = 0 if abs(res_val - mode_low) <= abs(res_val - mode_high) else 1

    bimodal_rows.append({
        "episode": r["episode"],
        "tau": r["tau"],
        "dim": dim,

        "mode_low": mode_low,
        "mode_high": mode_high,
        "mode_sep": sep,
        "sep_ratio": sep_ratio,
        "tol": tol,

        "te_val": te_val,
        "res_val": res_val,
        "te_in_gap": int(te_in_gap),
        "res_in_gap": int(res_in_gap),
        "te_in_mode": int(te_in_mode),
        "res_in_mode": int(res_in_mode),
        "te_nearest_mode": te_nearest_mode,
        "res_nearest_mode": res_nearest_mode,

        "res_minus_te": res_val - te_val,
        "abs_res_minus_te": abs(res_val - te_val),

        "weight_max": r["weight_max"],
        "weight_argmax": r["weight_argmax"],
        "effective_k": r["effective_k"],
        "entropy": r["entropy"],
        "delta_norm": r["delta_norm"],
        "kl_res_te": r["kl_res_te"],
        "output_shift_l2": r["output_shift_l2"],
    })

if analyzed_bimodal > 0:
    print(f"Analyzed bimodal τ: {analyzed_bimodal}")

    te_gap_rate = te_in_gap_count / analyzed_bimodal * 100
    res_gap_rate = res_in_gap_count / analyzed_bimodal * 100
    te_mode_rate = te_in_mode_count / analyzed_bimodal * 100
    res_mode_rate = res_in_mode_count / analyzed_bimodal * 100

    print("\nFixed TE output:")
    print(f"  in gap : {te_in_gap_count}/{analyzed_bimodal} ({te_gap_rate:.2f}%)")
    print(f"  in mode: {te_in_mode_count}/{analyzed_bimodal} ({te_mode_rate:.2f}%)")

    print("\nResidual output:")
    print(f"  in gap : {res_in_gap_count}/{analyzed_bimodal} ({res_gap_rate:.2f}%)")
    print(f"  in mode: {res_in_mode_count}/{analyzed_bimodal} ({res_mode_rate:.2f}%)")

    print("\nImprovement:")
    print(f"  gap reduction     : {te_gap_rate - res_gap_rate:.2f} percentage points")
    print(f"  mode rate increase: {res_mode_rate - te_mode_rate:.2f} percentage points")

    if res_gap_rate < te_gap_rate:
        print("  → Residual gap 비율이 TE보다 낮음: mode collapse 완화 신호 ✓")
    else:
        print("  → Residual gap 비율이 TE보다 낮지 않음: mode collapse 완화 증거 약함")

    bimodal_df = pd.DataFrame(bimodal_rows)
    bimodal_path = OUT_DIR / "mode_collapse_bimodal_metrics.csv"
    bimodal_df.to_csv(bimodal_path, index=False)
    print(f"\nSaved bimodal metrics: {bimodal_path}")

else:
    print("No valid bimodal decision points found.")
    print("Try lowering SPREAD_THRESHOLD or MIN_SEP_RATIO if this seems too strict.")


# ============================================================
# Visualization 1: delta / weight behavior
# ============================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

axes[0, 0].scatter(df["spread_max_dim"], df["delta_norm"], alpha=0.3, s=5)
axes[0, 0].axvline(SPREAD_THRESHOLD, color="red", linestyle="--")
axes[0, 0].set_xlabel("contribs spread max dim std")
axes[0, 0].set_ylabel("delta_norm")
axes[0, 0].set_title("Delta activation vs spread")
axes[0, 0].set_yscale("log")

axes[0, 1].scatter(df["spread_max_dim"], df["kl_res_te"], alpha=0.3, s=5)
axes[0, 1].axvline(SPREAD_THRESHOLD, color="red", linestyle="--")
axes[0, 1].set_xlabel("contribs spread max dim std")
axes[0, 1].set_ylabel("KL(res || TE)")
axes[0, 1].set_title("Residual weight deviation from TE")
axes[0, 1].set_yscale("log")

axes[0, 2].scatter(df["spread_max_dim"], df["output_shift_l2"], alpha=0.3, s=5)
axes[0, 2].axvline(SPREAD_THRESHOLD, color="red", linestyle="--")
axes[0, 2].set_xlabel("contribs spread max dim std")
axes[0, 2].set_ylabel("||res_output - te_output||")
axes[0, 2].set_title("Output shift from TE")
axes[0, 2].set_yscale("log")

axes[1, 0].hist(
    [df.loc[decision_mask, "delta_norm"], df.loc[~decision_mask, "delta_norm"]],
    bins=50,
    label=["decision τ", "non-decision τ"],
    alpha=0.7,
)
axes[1, 0].set_xlabel("delta_norm")
axes[1, 0].set_ylabel("count")
axes[1, 0].set_title("Delta norm distribution")
axes[1, 0].set_yscale("log")
axes[1, 0].legend()

axes[1, 1].hist(
    [df.loc[decision_mask, "effective_k"], df.loc[~decision_mask, "effective_k"]],
    bins=50,
    label=["decision τ", "non-decision τ"],
    alpha=0.7,
)
axes[1, 1].set_xlabel("effective_k")
axes[1, 1].set_ylabel("count")
axes[1, 1].set_title("Effective K distribution")
axes[1, 1].set_yscale("log")
axes[1, 1].legend()

axes[1, 2].hist(df["weight_argmax"], bins=min(L, 100))
axes[1, 2].set_xlabel("argmax candidate index")
axes[1, 2].set_ylabel("count")
axes[1, 2].set_title("Weight argmax distribution")

plt.tight_layout()
plot_path = OUT_DIR / "mode_collapse_summary.png"
plt.savefig(plot_path, dpi=140)
plt.close()
print(f"Saved: {plot_path}")


# ============================================================
# Visualization 2: top decision points
# ============================================================

top_decision_indices = np.argsort(spreads)[-5:][::-1]

fig, axes = plt.subplots(1, 5, figsize=(22, 4))

for ax_idx, r_idx in enumerate(top_decision_indices):
    r = results[int(r_idx)]
    dim = int(r["spread_max_dim_idx"])
    contribs_dim = r["contribs"][:, dim]

    valid, modes, sep, sep_ratio = analyze_bimodal_1d(contribs_dim)

    ax = axes[ax_idx]
    ax.hist(contribs_dim, bins=30, alpha=0.7)
    ax.axvline(r["te_output"][dim], color="red", linestyle="--", linewidth=2, label="Fixed TE")
    ax.axvline(r["res_output"][dim], color="green", linestyle="--", linewidth=2, label="Residual")

    title = f"τ={r['tau']}, ep={r['episode']}, dim={dim}\nspread={r['spread_max_dim']:.3f}"
    if valid:
        title += f", sep_ratio={sep_ratio:.2f}"

    ax.set_title(title)
    ax.set_xlabel("candidate action value")

    if ax_idx == 0:
        ax.legend()

plt.tight_layout()
plot_path = OUT_DIR / "top_decision_points.png"
plt.savefig(plot_path, dpi=140)
plt.close()
print(f"Saved: {plot_path}")


# ============================================================
# Visualization 3: sampled logits / weights
# ============================================================

if len(full_weight_rows) > 0:
    sample_df = pd.DataFrame(full_weight_rows)

    # spread 큰 샘플 3개만 그림
    sample_top = sample_df.sort_values("spread_max_dim", ascending=False).head(3)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    for row_idx, (_, row) in enumerate(sample_top.iterrows()):
        logits = np.array([row[f"logit_{i}"] for i in range(L)])
        weights = np.array([row[f"weight_{i}"] for i in range(L)])
        te_weights = np.array([row[f"te_weight_{i}"] for i in range(L)])

        axes[row_idx, 0].plot(logits)
        axes[row_idx, 0].set_title(f"Logits | ep={int(row['episode'])}, τ={int(row['tau'])}")
        axes[row_idx, 0].set_xlabel("candidate index")
        axes[row_idx, 0].set_ylabel("logit")

        axes[row_idx, 1].plot(weights, label="Residual weight")
        axes[row_idx, 1].plot(te_weights, linestyle="--", label="TE weight")
        axes[row_idx, 1].set_title(
            f"Weights | argmax={int(row['weight_argmax'])}, "
            f"effK={row['effective_k']:.2f}"
        )
        axes[row_idx, 1].set_xlabel("candidate index")
        axes[row_idx, 1].set_ylabel("weight")
        axes[row_idx, 1].legend()

    plt.tight_layout()
    plot_path = OUT_DIR / "sampled_logits_weights.png"
    plt.savefig(plot_path, dpi=140)
    plt.close()
    print(f"Saved: {plot_path}")


# ============================================================
# Final guide
# ============================================================

print("\n" + "=" * 80)
print("해석 가이드")
print("=" * 80)

print("""
1. Delta activation
   - decision/non-decision delta ratio > 1.5
     → decision point에서 residual delta가 더 활성화됨.
   - ratio ≈ 1.0
     → decision 여부와 상관없이 비슷하게 동작. mode-aware 증거 약함.

2. Weight behavior
   - decision에서 KL(res||TE), output_shift_l2가 더 크면
     → decision point에서 TE prior에서 더 많이 벗어남.
   - effective_k가 L에 가까우면
     → 거의 uniform average에 가까움.
   - effective_k가 작고 weight_max가 크면
     → 후보를 선택적으로 고름.

3. Bimodal decision analysis
   - TE in gap 높고, Residual in gap 낮으면
     → Fixed TE의 mode collapse를 residual이 완화한 증거.
   - Residual in mode가 TE보다 높으면
     → residual이 gap 평균이 아니라 한 mode 쪽으로 이동한 증거.

4. 그림 확인
   - top_decision_points.png:
     빨간선 Fixed TE가 두 봉우리 사이에 있고,
     초록선 Residual이 한 봉우리 근처면 가장 좋은 시각적 증거.

5. 주의
   - 이건 offline cache 검증이라서 success 52%의 원인을 완전히 증명하진 않음.
   - 하지만 mode collapse 완화 여부를 보는 1차 검증으로는 충분히 좋음.
""")

print("\nDone.")
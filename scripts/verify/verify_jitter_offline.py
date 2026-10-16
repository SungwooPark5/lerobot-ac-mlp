"""
Offline jitter 검증:
- cache chunks + trained residual weight network 사용
- rollout 없이 episode별 action sequence 생성
- latest/current vs fixed TE vs residual learned ensemble 비교

출력:
  1) jitter scalar metrics csv
  2) per-step metrics csv
  3) summary plots
  4) example action trajectory plots
"""

import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths / Config
# ============================================================

CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
WEIGHT_NET_PATH = Path("/home1/eunji24/lerobot_project/outputs/weight_network_residual_lam5/weight_net_final.pt")

OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/jitter_verify_lam5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# action dim별로 너무 큰 값 때문에 평균이 흔들리면 True로 바꿔도 됨
NORMALIZE_BY_STD = False

# 그림으로 볼 episode 수
N_PLOT_EPISODES = 5


# ============================================================
# Model class: Stage 2와 동일
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

        state_tok = self.state_proj(state).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)

        memory = torch.cat([state_tok, chunk_toks], dim=1)

        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)

        delta = self.head(x).squeeze(-1)
        final_logits = self.te_logits.unsqueeze(0) + delta

        return final_logits, delta


# ============================================================
# Metric utils
# ============================================================

def safe_mean(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return float(np.mean(x))


def safe_median(x):
    x = np.asarray(x)
    if len(x) == 0:
        return np.nan
    return float(np.median(x))


def l1_per_step(x):
    """
    x: (T, action_dim)
    return: (T-1,)
    """
    dx = np.diff(x, axis=0)
    return np.mean(np.abs(dx), axis=1)


def l2_per_step(x):
    dx = np.diff(x, axis=0)
    return np.linalg.norm(dx, axis=1)


def accel_l1_per_step(x):
    """
    x_t - 2x_{t-1} + x_{t-2}
    return: (T-2,)
    """
    ddx = x[2:] - 2 * x[1:-1] + x[:-2]
    return np.mean(np.abs(ddx), axis=1)


def accel_l2_per_step(x):
    ddx = x[2:] - 2 * x[1:-1] + x[:-2]
    return np.linalg.norm(ddx, axis=1)


def jerk_l2_per_step(x):
    """
    third difference
    return: (T-3,)
    """
    dddx = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
    return np.linalg.norm(dddx, axis=1)


def direction_flip_rate(x, eps=1e-6):
    """
    action 변화량의 sign이 얼마나 자주 바뀌는지.
    dim별 flip 비율 평균.
    """
    dx = np.diff(x, axis=0)  # (T-1, D)
    if dx.shape[0] < 2:
        return np.nan

    sign = np.sign(dx)
    valid = np.abs(dx) > eps

    flips = (sign[1:] * sign[:-1]) < 0
    valid_pair = valid[1:] & valid[:-1]

    if valid_pair.sum() == 0:
        return np.nan

    return float(flips[valid_pair].mean())


def high_freq_energy_ratio(x):
    """
    FFT 기반 고주파 에너지 비율.
    낮을수록 부드럽다고 볼 수 있음.
    각 action dim별로 계산 후 평균.
    """
    T, D = x.shape
    if T < 8:
        return np.nan

    ratios = []

    for d in range(D):
        y = x[:, d] - x[:, d].mean()
        fft = np.fft.rfft(y)
        power = np.abs(fft) ** 2

        if power.sum() < 1e-12:
            continue

        freqs = np.fft.rfftfreq(T)

        # normalized frequency 기준 상위 1/3을 high-frequency로 둠
        high_mask = freqs > (freqs.max() * 2.0 / 3.0)
        ratio = power[high_mask].sum() / power.sum()
        ratios.append(ratio)

    if len(ratios) == 0:
        return np.nan

    return float(np.mean(ratios))


def summarize_sequence(name, seq):
    """
    seq: (T, action_dim)
    """
    jit_l1 = l1_per_step(seq)
    jit_l2 = l2_per_step(seq)
    acc_l1 = accel_l1_per_step(seq)
    acc_l2 = accel_l2_per_step(seq)
    jerk_l2 = jerk_l2_per_step(seq)

    return {
        "method": name,
        "T": seq.shape[0],

        "jitter_l1_mean": safe_mean(jit_l1),
        "jitter_l1_median": safe_median(jit_l1),
        "jitter_l2_mean": safe_mean(jit_l2),
        "jitter_l2_median": safe_median(jit_l2),

        "accel_l1_mean": safe_mean(acc_l1),
        "accel_l1_median": safe_median(acc_l1),
        "accel_l2_mean": safe_mean(acc_l2),
        "accel_l2_median": safe_median(acc_l2),

        "jerk_l2_mean": safe_mean(jerk_l2),
        "jerk_l2_median": safe_median(jerk_l2),

        "direction_flip_rate": direction_flip_rate(seq),
        "high_freq_energy_ratio": high_freq_energy_ratio(seq),
    }


def per_step_metrics(episode, method, seq):
    """
    step별 jitter/accel 저장
    """
    rows = []

    jit_l1 = l1_per_step(seq)
    jit_l2 = l2_per_step(seq)
    acc_l1 = accel_l1_per_step(seq)
    acc_l2 = accel_l2_per_step(seq)

    T = seq.shape[0]

    for i in range(T):
        row = {
            "episode": episode,
            "method": method,
            "local_step": i,
        }

        if i >= 1:
            row["jitter_l1"] = float(jit_l1[i - 1])
            row["jitter_l2"] = float(jit_l2[i - 1])
        else:
            row["jitter_l1"] = np.nan
            row["jitter_l2"] = np.nan

        if i >= 2:
            row["accel_l1"] = float(acc_l1[i - 2])
            row["accel_l2"] = float(acc_l2[i - 2])
        else:
            row["accel_l1"] = np.nan
            row["accel_l2"] = np.nan

        rows.append(row)

    return rows


# ============================================================
# Load
# ============================================================

print("=" * 80)
print("Offline Jitter Verify")
print("=" * 80)

print("\nLoading cache...")
cache = np.load(CACHE_PATH)

chunks = cache["chunks"]
states = cache["states"]
ep_starts = cache["ep_starts"]

N, chunk_size, action_dim = chunks.shape
state_dim = states.shape[1]

print(f"  chunks    : {chunks.shape}")
print(f"  states    : {states.shape}")
print(f"  ep_starts : {ep_starts.shape}")
print(f"  action_dim={action_dim}, state_dim={state_dim}")

print(f"\nLoading weight network from {WEIGHT_NET_PATH}")
ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)

L = int(ckpt["L"])
te_coeff = float(ckpt["te_coeff"])

model = ResidualWeightNetwork(
    L=L,
    state_dim=int(ckpt["state_dim"]),
    action_dim=int(ckpt["action_dim"]),
    te_coeff=te_coeff,
    d_model=int(ckpt["d_model"]),
    n_heads=int(ckpt["n_heads"]),
    n_layers=int(ckpt["n_layers"]),
).to(DEVICE)

model.load_state_dict(ckpt["model"])
model.eval()

print(f"  L               : {L}")
print(f"  te_coeff        : {te_coeff}")
print(f"  lambda_delta_reg: {ckpt.get('lambda_delta_reg', '?')}")

if L > chunk_size:
    raise ValueError(f"L={L} > chunk_size={chunk_size}")

# TE weights
te_weights = np.exp(-te_coeff * np.arange(L))
te_weights = te_weights / te_weights.sum()

print("\nIndex meaning:")
print("  idx 0   = newest/current candidate")
print("  idx L-1 = oldest candidate")


# ============================================================
# Build offline action sequences
# ============================================================

print("\nBuilding offline action sequences...")

all_summary_rows = []
all_step_rows = []

plot_data = []

for ep in range(len(ep_starts)):
    ep_from = int(ep_starts[ep][0])
    ep_to = int(ep_starts[ep][1])

    if ep_to - ep_from < L:
        continue

    latest_seq = []
    te_seq = []
    residual_seq = []
    tau_list = []

    weight_max_list = []
    effective_k_list = []
    argmax_list = []

    for win_start in range(ep_from, ep_to - L + 1):
        tau = win_start + L - 1

        frame_idx = tau - np.arange(L)
        pos_idx = np.arange(L)

        contribs_np = chunks[frame_idx, pos_idx]  # (L, action_dim)
        state_np = states[tau]

        # latest/current action: idx 0
        latest_action = contribs_np[0]

        # fixed TE action
        te_action = (te_weights[:, None] * contribs_np).sum(axis=0)

        # residual action
        state_t = torch.from_numpy(state_np).float().to(DEVICE).unsqueeze(0)
        contribs_t = torch.from_numpy(contribs_np).float().to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            final_logits, delta = model(state_t, contribs_t)
            weights = F.softmax(final_logits, dim=1).squeeze(0).cpu().numpy()

        residual_action = (weights[:, None] * contribs_np).sum(axis=0)

        latest_seq.append(latest_action)
        te_seq.append(te_action)
        residual_seq.append(residual_action)
        tau_list.append(tau)

        weight_max_list.append(float(weights.max()))
        effective_k_list.append(float(1.0 / (np.sum(weights ** 2) + 1e-8)))
        argmax_list.append(int(weights.argmax()))

    latest_seq = np.asarray(latest_seq)
    te_seq = np.asarray(te_seq)
    residual_seq = np.asarray(residual_seq)

    if NORMALIZE_BY_STD:
        # 세 sequence 전체 기준으로 dim별 std 정규화
        concat = np.concatenate([latest_seq, te_seq, residual_seq], axis=0)
        std = concat.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        latest_seq = latest_seq / std
        te_seq = te_seq / std
        residual_seq = residual_seq / std

    seqs = {
        "latest": latest_seq,
        "fixed_te": te_seq,
        "residual": residual_seq,
    }

    for method, seq in seqs.items():
        s = summarize_sequence(method, seq)
        s["episode"] = ep
        s["global_tau_start"] = tau_list[0]
        s["global_tau_end"] = tau_list[-1]
        all_summary_rows.append(s)

        all_step_rows.extend(per_step_metrics(ep, method, seq))

    if len(plot_data) < N_PLOT_EPISODES:
        plot_data.append({
            "episode": ep,
            "tau": np.asarray(tau_list),
            "latest": latest_seq,
            "fixed_te": te_seq,
            "residual": residual_seq,
            "weight_max": np.asarray(weight_max_list),
            "effective_k": np.asarray(effective_k_list),
            "argmax": np.asarray(argmax_list),
        })

print("  Done.")


# ============================================================
# Save CSV
# ============================================================

summary_df = pd.DataFrame(all_summary_rows)
step_df = pd.DataFrame(all_step_rows)

summary_path = OUT_DIR / "jitter_summary_by_episode.csv"
step_path = OUT_DIR / "jitter_per_step.csv"

summary_df.to_csv(summary_path, index=False)
step_df.to_csv(step_path, index=False)

print(f"\nSaved summary: {summary_path}")
print(f"Saved per-step: {step_path}")


# ============================================================
# Aggregate report
# ============================================================

print("\n" + "=" * 80)
print("Aggregate jitter report")
print("=" * 80)

metrics = [
    "jitter_l1_mean",
    "jitter_l2_mean",
    "accel_l1_mean",
    "accel_l2_mean",
    "jerk_l2_mean",
    "direction_flip_rate",
    "high_freq_energy_ratio",
]

agg = summary_df.groupby("method")[metrics].agg(["mean", "median"])
print(agg)

agg_path = OUT_DIR / "jitter_aggregate_metrics.csv"
agg.to_csv(agg_path)
print(f"\nSaved aggregate: {agg_path}")


def pct_change(a, b):
    """
    b relative to a.
    negative means b decreased.
    """
    return (b - a) / max(abs(a), 1e-9) * 100.0


print("\n" + "=" * 80)
print("Relative change")
print("=" * 80)

method_mean = summary_df.groupby("method")[metrics].mean()

if "latest" in method_mean.index and "fixed_te" in method_mean.index and "residual" in method_mean.index:
    for m in metrics:
        latest_val = method_mean.loc["latest", m]
        te_val = method_mean.loc["fixed_te", m]
        res_val = method_mean.loc["residual", m]

        print(f"\n[{m}]")
        print(f"  latest    : {latest_val:.6f}")
        print(f"  fixed_te  : {te_val:.6f}  ({pct_change(latest_val, te_val):+.2f}% vs latest)")
        print(f"  residual  : {res_val:.6f}  ({pct_change(latest_val, res_val):+.2f}% vs latest)")
        print(f"  residual vs fixed_te: {pct_change(te_val, res_val):+.2f}%")

print("\n해석:")
print("  - residual jitter/accel이 latest보다 낮으면: 떨림 완화")
print("  - residual이 fixed_te보다 높으면: TE보다 덜 부드럽지만, mode 선택 보정 때문에 그럴 수 있음")
print("  - residual이 latest보다도 높으면: 보정이 떨림을 만든 것일 수 있음")
print("  - accel/jerk가 중요함. 단순 jitter보다 acceleration이 떨림 체감에 더 가까움.")


# ============================================================
# Plots: aggregate bar plots
# ============================================================

print("\nGenerating plots...")

for metric in metrics:
    plt.figure(figsize=(7, 4))

    vals = method_mean.loc[["latest", "fixed_te", "residual"], metric]
    plt.bar(vals.index, vals.values)

    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()

    out = OUT_DIR / f"{metric}_bar.png"
    plt.savefig(out, dpi=140)
    plt.close()

print("Saved bar plots.")


# ============================================================
# Plots: example trajectories
# ============================================================

# action dim 중 variation 큰 dim 위주로 그림
for item in plot_data:
    ep = item["episode"]
    tau = item["tau"]

    # residual 기준으로 변화량 큰 dim 선택
    variation = item["residual"].std(axis=0)
    top_dims = np.argsort(variation)[-4:][::-1]

    fig, axes = plt.subplots(len(top_dims), 1, figsize=(12, 3 * len(top_dims)))

    if len(top_dims) == 1:
        axes = [axes]

    for ax, d in zip(axes, top_dims):
        ax.plot(tau, item["latest"][:, d], label="latest", alpha=0.8)
        ax.plot(tau, item["fixed_te"][:, d], label="fixed_te", alpha=0.8)
        ax.plot(tau, item["residual"][:, d], label="residual", alpha=0.8)
        ax.set_title(f"Episode {ep} | action dim {d}")
        ax.set_xlabel("tau")
        ax.set_ylabel("action value")
        ax.legend()

    plt.tight_layout()
    out = OUT_DIR / f"episode_{ep}_action_trajectories.png"
    plt.savefig(out, dpi=140)
    plt.close()

# weight behavior plot
for item in plot_data:
    ep = item["episode"]
    tau = item["tau"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(tau, item["weight_max"])
    axes[0].set_title(f"Episode {ep} | residual weight max")
    axes[0].set_xlabel("tau")
    axes[0].set_ylabel("weight_max")

    axes[1].plot(tau, item["effective_k"])
    axes[1].set_title(f"Episode {ep} | effective_k")
    axes[1].set_xlabel("tau")
    axes[1].set_ylabel("effective_k")

    axes[2].plot(tau, item["argmax"])
    axes[2].set_title(f"Episode {ep} | argmax candidate index")
    axes[2].set_xlabel("tau")
    axes[2].set_ylabel("argmax idx")

    plt.tight_layout()
    out = OUT_DIR / f"episode_{ep}_weight_behavior.png"
    plt.savefig(out, dpi=140)
    plt.close()

print("Saved trajectory plots.")


# ============================================================
# Additional: per-step jitter distribution plot
# ============================================================

for metric in ["jitter_l2", "accel_l2"]:
    plt.figure(figsize=(8, 4))

    for method in ["latest", "fixed_te", "residual"]:
        vals = step_df.loc[step_df["method"] == method, metric].dropna().to_numpy()
        if len(vals) == 0:
            continue

        # extreme tail 때문에 99 percentile까지만 표시
        hi = np.percentile(vals, 99)
        vals_clip = vals[vals <= hi]

        plt.hist(vals_clip, bins=80, alpha=0.5, label=method)

    plt.xlabel(metric)
    plt.ylabel("count")
    plt.title(f"{metric} distribution, clipped at each method 99 percentile")
    plt.legend()
    plt.tight_layout()

    out = OUT_DIR / f"{metric}_distribution.png"
    plt.savefig(out, dpi=140)
    plt.close()

print("Saved distribution plots.")

print("\nDone.")
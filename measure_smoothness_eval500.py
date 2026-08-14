import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOTS = {
    "act": "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/final_act_K100_m100_100k_eval500",
    "acm": "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/final_acm_mamba_baseline_K100_m100_100k_eval500",
    "bimamba": "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/final_bimamba_K100_m100_100k_eval500_seed1000",
    "selfattn": "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_selfattn_only_100k_eval500",
    "pre_crossattn": "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_pre_crossattn_K100_m100_100k_eval500",

    "acm2": "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm2_mamba_K100_m100_100k_eval500",
    "acm3": "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm3_K100_m100_100k_eval500_b1",
}

OUT_CSV = "smoothness_eval500_summary.csv"


def find_action_logs(root):
    root = Path(root)
    return sorted(root.rglob("action_logs"))


def load_action_file(path):
    path = Path(path)

    if path.suffix in [".pt", ".pth"]:
        import torch
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict) and "actions" in data:
            return data["actions"].detach().cpu().numpy()
        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()

        return np.asarray(data)

    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".npz":
        data = np.load(path)
        for k in ["actions", "action", "pred_actions", "raw_actions"]:
            if k in data:
                return data[k]
        return data[list(data.keys())[0]]

    if path.suffix == ".csv":
        return pd.read_csv(path).values

    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        return np.asarray(data)

    return None


def collect_action_arrays(log_dir):
    log_dir = Path(log_dir)
    arrays = []

    for p in sorted(log_dir.rglob("*")):
        if p.suffix not in [".pt", ".pth", ".npy", ".npz", ".csv", ".json"]:
            continue

        try:
            arr = load_action_file(p)
        except Exception:
            continue

        if arr is None:
            continue

        arr = np.asarray(arr, dtype=np.float64)

        # expected: [T, action_dim] or [episode, T, action_dim]
        if arr.ndim == 2 and arr.shape[-1] <= 64:
            arrays.append(arr)
        elif arr.ndim == 3 and arr.shape[-1] <= 64:
            for i in range(arr.shape[0]):
                arrays.append(arr[i])

    return arrays


def safe_norm(x, axis=-1):
    return np.linalg.norm(x, axis=axis)


def log_dimensionless_jerk(x, eps=1e-12):
    """
    LDLJ-style metric.
    Higher negative magnitude generally means less smooth.
    We report abs_ldj = -ldj so smaller is smoother-like.
    """
    T = x.shape[0]
    if T < 4:
        return np.nan

    vel = np.diff(x, axis=0)
    speed = safe_norm(vel, axis=-1)
    amplitude = np.sum(speed) + eps

    jerk = np.diff(x, n=3, axis=0)
    jerk_sq_sum = np.sum(jerk ** 2)

    duration = float(T - 1)
    val = (duration ** 5 / (amplitude ** 2 + eps)) * jerk_sq_sum + eps
    ldj = -np.log(val)
    return ldj


def sparc_1d(signal, fs=1.0, padlevel=4, fc=10.0, amp_th=0.05, eps=1e-12):
    """
    Spectral Arc Length.
    More negative = less smooth.
    Closer to 0 = smoother.
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)

    n = len(signal)
    if n < 8:
        return np.nan

    nfft = int(2 ** np.ceil(np.log2(n)) * (2 ** padlevel))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(signal, n=nfft))

    if np.max(mag) < eps:
        return 0.0

    mag = mag / (np.max(mag) + eps)

    valid = freqs <= fc
    freqs = freqs[valid]
    mag = mag[valid]

    idx = np.where(mag >= amp_th)[0]
    if len(idx) < 2:
        return 0.0

    last = idx[-1] + 1
    freqs = freqs[:last]
    mag = mag[:last]

    f_norm = freqs / (freqs[-1] + eps)
    dm = np.diff(mag)
    df = np.diff(f_norm)

    sal = -np.sum(np.sqrt(df ** 2 + dm ** 2))
    return sal


def sparc_multidim(x):
    vals = []
    for d in range(x.shape[1]):
        vals.append(sparc_1d(x[:, d]))
    return np.nanmean(vals)


def metrics_for_traj(x):
    x = np.asarray(x, dtype=np.float64)

    if x.ndim != 2 or x.shape[0] < 4:
        return None

    diff = np.diff(x, n=1, axis=0)
    accel = np.diff(x, n=2, axis=0)
    jerk = np.diff(x, n=3, axis=0)

    jerk_l2 = safe_norm(jerk, axis=-1)

    # sign flip: adjacent velocity direction changes per action dimension
    v = diff
    flip = (v[1:] * v[:-1]) < 0

    return {
        "T": x.shape[0],
        "D": x.shape[1],

        "diff_abs_mean": np.mean(np.abs(diff)),
        "diff_l2_mean": np.mean(safe_norm(diff, axis=-1)),

        "accel_abs_mean": np.mean(np.abs(accel)),
        "accel_l2_mean": np.mean(safe_norm(accel, axis=-1)),

        "jerk_abs_mean": np.mean(np.abs(jerk)),
        "jerk_l2_mean": np.mean(jerk_l2),
        "jerk_rms": np.sqrt(np.mean(jerk ** 2)),
        "jerk_l2_rms": np.sqrt(np.mean(jerk_l2 ** 2)),
        "jerk_p95": np.percentile(np.abs(jerk), 95),
        "jerk_l2_p95": np.percentile(jerk_l2, 95),

        "ldj": log_dimensionless_jerk(x),
        "ldj_abs": -log_dimensionless_jerk(x),

        "sparc": sparc_multidim(x),

        "sign_flip_mean": np.mean(flip),
    }


def summarize_model(name, root):
    log_dirs = find_action_logs(root)

    if not log_dirs:
        print(f"[WARN] no action_logs found: {name} {root}")
        return None

    all_arrays = []
    for d in log_dirs:
        arrs = collect_action_arrays(d)
        all_arrays.extend(arrs)

    if not all_arrays:
        print(f"[WARN] no action arrays found: {name}")
        return None

    rows = []
    for arr in all_arrays:
        m = metrics_for_traj(arr)
        if m is not None:
            rows.append(m)

    if not rows:
        print(f"[WARN] no valid trajectories: {name}")
        return None

    df = pd.DataFrame(rows)

    out = {
        "model": name,
        "n_traj": len(df),
        "T_mean": df["T"].mean(),
        "D_mean": df["D"].mean(),
    }

    metric_cols = [c for c in df.columns if c not in ["T", "D"]]
    for c in metric_cols:
        out[c + "_mean"] = df[c].mean()
        out[c + "_std"] = df[c].std()

    return out


def main():
    summaries = []

    for name, root in ROOTS.items():
        print(f"\n===== {name} =====")
        s = summarize_model(name, root)
        if s is not None:
            summaries.append(s)
            print(pd.Series(s).to_string())

    df = pd.DataFrame(summaries)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    cols = [
        "model", "n_traj",
        "diff_abs_mean_mean",
        "accel_abs_mean_mean",
        "jerk_abs_mean_mean",
        "jerk_rms_mean",
        "jerk_l2_rms_mean",
        "jerk_p95_mean",
        "ldj_abs_mean",
        "sparc_mean",
        "sign_flip_mean_mean",
    ]
    cols = [c for c in cols if c in df.columns]
    print("\n===== compact summary =====")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
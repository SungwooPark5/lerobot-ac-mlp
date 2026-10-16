import json
import torch
import numpy as np
from pathlib import Path

ROOT = Path("outputs/eval/smoothness_100k")

RUNS = [
    "act_transfer_cube_test_100k",
    "acm_transfer_cube_test_100k",
    "acm_selfattn_front_accel0p001_100k",
    "acm_refiner_k7_front_accel0p001_100k",
    "acm_mamba_gated_action_selfattn_100k",
    "acm_bimamba_decoder_100k_seed1000",
]

def metrics_for_actions(a):
    a = a.astype(np.float64)

    diff = a[1:] - a[:-1]
    accel = a[2:] - 2*a[1:-1] + a[:-2]
    jerk = a[3:] - 3*a[2:-1] + 3*a[1:-2] - a[:-3]

    sign_flip = ((diff[1:] * diff[:-1]) < 0).mean()

    return {
        "diff_l2": np.linalg.norm(diff, axis=-1).mean(),
        "diff_abs": np.abs(diff).mean(),
        "accel_l2": np.linalg.norm(accel, axis=-1).mean(),
        "accel_abs": np.abs(accel).mean(),
        "jerk_l2": np.linalg.norm(jerk, axis=-1).mean(),
        "jerk_abs": np.abs(jerk).mean(),
        "sign_flip": sign_flip,
    }

all_results = {}

for run in RUNS:
    log_dir = ROOT / run / "videos" / "action_logs"
    files = sorted(log_dir.glob("episode_*.pt"))

    episode_metrics = []
    for f in files:
        d = torch.load(f, map_location="cpu")
        a = d["actions"].numpy()  # (400,14)
        episode_metrics.append(metrics_for_actions(a))

    summary = {}
    for k in episode_metrics[0].keys():
        vals = np.array([m[k] for m in episode_metrics])
        summary[k + "_mean"] = float(vals.mean())
        summary[k + "_std"] = float(vals.std())

    summary["n_episodes"] = len(files)
    all_results[run] = summary

print("\n=== Smoothness Metrics from Executed Actions ===")
print("model,diff_l2,accel_l2,jerk_l2,jerk_abs,sign_flip")
for run, m in all_results.items():
    print(
        f"{run},"
        f"{m['diff_l2_mean']:.6f},"
        f"{m['accel_l2_mean']:.6f},"
        f"{m['jerk_l2_mean']:.6f},"
        f"{m['jerk_abs_mean']:.6f},"
        f"{m['sign_flip_mean']:.6f}"
    )

out = ROOT / "smoothness_metrics_100k.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n[SAVE] {out}")
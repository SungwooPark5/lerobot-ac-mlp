#!/usr/bin/env python
import numpy as np
import torch
from pathlib import Path

# 필요하면 경로 수정
ACT_CACHE = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
ACM_CACHE = Path("/home1/eunji24/lerobot_project/cache/acm_60k_chunks_vis/chunks.npz")

L = 100


def load_cache(path):
    print(f"\n[LOAD] {path}")
    c = np.load(path)
    print("keys:", c.files)

    chunks = torch.from_numpy(c["chunks"].copy()).float()   # [N, L, A]
    ep_starts = c["ep_starts"].astype(np.int64)

    print("chunks:", tuple(chunks.shape))
    print("episodes:", len(ep_starts))
    return chunks, ep_starts


def analyze(chunks, ep_starts, name):
    overlap_vals = []
    vel_vals = []
    acc_vals = []
    jerk_vals = []

    for s, e in ep_starts:
        ep_chunks = chunks[s:e]  # [T, L, A]

        if ep_chunks.shape[0] < 2:
            continue

        # timestep 간 overlap consistency
        # chunk_t[1:] vs chunk_{t+1}[:-1]
        overlap = ep_chunks[:-1, 1:, :] - ep_chunks[1:, :-1, :]
        overlap_l2 = torch.norm(overlap, dim=-1)  # [T-1, L-1]
        overlap_vals.append(overlap_l2)

        # chunk 내부 smoothness
        vel = ep_chunks[:, 1:, :] - ep_chunks[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        jerk = acc[:, 1:, :] - acc[:, :-1, :]

        vel_vals.append(torch.norm(vel, dim=-1))
        acc_vals.append(torch.norm(acc, dim=-1))
        jerk_vals.append(torch.norm(jerk, dim=-1))

    overlap_vals = torch.cat([x.flatten() for x in overlap_vals])
    vel_vals = torch.cat([x.flatten() for x in vel_vals])
    acc_vals = torch.cat([x.flatten() for x in acc_vals])
    jerk_vals = torch.cat([x.flatten() for x in jerk_vals])

    stats = {
        "overlap_mean": overlap_vals.mean().item(),
        "overlap_median": overlap_vals.median().item(),
        "overlap_p90": overlap_vals.quantile(0.90).item(),
        "overlap_p95": overlap_vals.quantile(0.95).item(),
        "overlap_max": overlap_vals.max().item(),

        "chunk_vel_mean": vel_vals.mean().item(),
        "chunk_vel_p95": vel_vals.quantile(0.95).item(),

        "chunk_acc_mean": acc_vals.mean().item(),
        "chunk_acc_p95": acc_vals.quantile(0.95).item(),

        "chunk_jerk_mean": jerk_vals.mean().item(),
        "chunk_jerk_p95": jerk_vals.quantile(0.95).item(),
    }

    print(f"\n[{name}]")
    for k, v in stats.items():
        print(f"{k:20s}: {v:.6f}")

    return stats


def compare(act, acm):
    print("\n" + "=" * 100)
    print("ACT vs ACM stability comparison")
    print("=" * 100)
    print(f"{'metric':22s} | {'ACT':>12s} | {'ACM':>12s} | {'diff':>12s} | {'ratio':>8s} | {'% increase':>10s}")
    print("-" * 100)

    for k in act.keys():
        a = act[k]
        m = acm[k]
        diff = m - a
        ratio = m / a if abs(a) > 1e-12 else float("nan")
        pct = (ratio - 1.0) * 100.0 if abs(a) > 1e-12 else float("nan")
        print(f"{k:22s} | {a:12.6f} | {m:12.6f} | {diff:12.6f} | {ratio:8.3f} | {pct:9.1f}%")

    print("=" * 100)


def main():
    act_chunks, act_ep = load_cache(ACT_CACHE)
    acm_chunks, acm_ep = load_cache(ACM_CACHE)

    act_stats = analyze(act_chunks, act_ep, "ACT")
    acm_stats = analyze(acm_chunks, acm_ep, "ACM")

    compare(act_stats, acm_stats)


if __name__ == "__main__":
    main()
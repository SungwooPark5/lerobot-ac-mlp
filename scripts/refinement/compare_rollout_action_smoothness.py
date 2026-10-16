#!/usr/bin/env python

import argparse
from pathlib import Path
import glob

import torch
import numpy as np


def load_actions(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu")

    if "actions" not in data:
        raise KeyError(f"{path} does not contain key 'actions'. Available keys: {list(data.keys())}")

    actions = data["actions"].float()

    # Common shapes:
    # (T, A)
    # (T, 1, A)
    # (T, B, A)
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = actions.squeeze(1)

    if actions.ndim != 2:
        raise ValueError(f"Expected actions shape (T, A), got {tuple(actions.shape)} from {path}")

    return actions


def l2_mean(x: torch.Tensor) -> float:
    return torch.linalg.norm(x, dim=-1).mean().item()


def l2_max(x: torch.Tensor) -> float:
    return torch.linalg.norm(x, dim=-1).max().item()


def abs_mean(x: torch.Tensor) -> float:
    return x.abs().mean().item()


def per_dim_abs_mean(x: torch.Tensor) -> np.ndarray:
    return x.abs().mean(dim=0).cpu().numpy()


def diff_seq(actions: torch.Tensor) -> torch.Tensor:
    return actions[1:] - actions[:-1]


def accel_seq(actions: torch.Tensor) -> torch.Tensor:
    return actions[2:] - 2.0 * actions[1:-1] + actions[:-2]


def jerk_seq(actions: torch.Tensor) -> torch.Tensor:
    return (
        actions[3:]
        - 3.0 * actions[2:-1]
        + 3.0 * actions[1:-2]
        - actions[:-3]
    )


def boundary_jump(actions: torch.Tensor, n_action_steps: int) -> torch.Tensor | None:
    vals = []
    T = actions.shape[0]

    for b in range(n_action_steps, T, n_action_steps):
        vals.append(torch.linalg.norm(actions[b] - actions[b - 1], dim=-1))

    if len(vals) == 0:
        return None

    return torch.stack(vals)


def sign_flip_rate(actions: torch.Tensor, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """
    Velocity sign flip rate.
    actions: (T, A)
    """
    diff = actions[1:] - actions[:-1]  # (T-1, A)

    if diff.shape[0] < 2:
        per_dim = np.zeros(actions.shape[-1], dtype=np.float32)
        return per_dim, 0.0

    d0 = diff[:-1]
    d1 = diff[1:]

    valid = (d0.abs() > eps) & (d1.abs() > eps)
    flips = (d0 * d1 < 0) & valid

    denom = valid.sum(dim=0).clamp(min=1)
    per_dim = flips.sum(dim=0).float() / denom.float()

    return per_dim.cpu().numpy(), per_dim.mean().item()


def metrics_for_episode(actions: torch.Tensor, n_action_steps: int) -> dict:
    diff = diff_seq(actions)
    accel = accel_seq(actions)
    jerk = jerk_seq(actions)
    bj = boundary_jump(actions, n_action_steps)
    flip_per_dim, flip_mean = sign_flip_rate(actions)

    out = {
        "T": actions.shape[0],
        "A": actions.shape[1],

        "diff_l2_mean": l2_mean(diff),
        "diff_l2_max": l2_max(diff),
        "diff_abs_mean": abs_mean(diff),

        "accel_l2_mean": l2_mean(accel),
        "accel_l2_max": l2_max(accel),
        "accel_abs_mean": abs_mean(accel),

        "jerk_l2_mean": l2_mean(jerk),
        "jerk_l2_max": l2_max(jerk),
        "jerk_abs_mean": abs_mean(jerk),

        "sign_flip_mean": flip_mean,

        "diff_abs_per_dim": per_dim_abs_mean(diff),
        "accel_abs_per_dim": per_dim_abs_mean(accel),
        "jerk_abs_per_dim": per_dim_abs_mean(jerk),
        "sign_flip_per_dim": flip_per_dim,
    }

    if bj is not None:
        out["boundary_jump_l2_mean"] = bj.mean().item()
        out["boundary_jump_l2_max"] = bj.max().item()
        out["num_boundaries"] = int(bj.numel())
    else:
        out["boundary_jump_l2_mean"] = np.nan
        out["boundary_jump_l2_max"] = np.nan
        out["num_boundaries"] = 0

    return out


def find_episode_files(path: str) -> list[Path]:
    p = Path(path)

    if p.is_file():
        return [p]

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    files = sorted(p.glob("episode_*.pt"))
    if len(files) == 0:
        files = sorted(p.glob("*.pt"))

    if len(files) == 0:
        raise FileNotFoundError(f"No .pt files found in {p}")

    return files


def mean_scalar(metrics: list[dict], key: str) -> float:
    vals = [m[key] for m in metrics if key in m and not np.isnan(m[key])]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def mean_array(metrics: list[dict], key: str) -> np.ndarray | None:
    vals = [m[key] for m in metrics if key in m]
    if len(vals) == 0:
        return None
    return np.mean(vals, axis=0)


def summarize_group(name: str, files: list[Path], n_action_steps: int, max_episodes: int | None) -> tuple[list[dict], dict]:
    if max_episodes is not None:
        files = files[:max_episodes]

    print("\n" + "=" * 100)
    print(f"[{name}]")
    print(f"num files = {len(files)}")
    print("=" * 100)

    metrics = []

    for f in files:
        actions = load_actions(f)
        m = metrics_for_episode(actions, n_action_steps)
        metrics.append(m)

        print(
            f"{f.name:20s} | "
            f"T={m['T']:4d} | "
            f"diff={m['diff_l2_mean']:.6f} | "
            f"accel={m['accel_l2_mean']:.6f} | "
            f"jerk={m['jerk_l2_mean']:.6f} | "
            f"boundary={m['boundary_jump_l2_mean']:.6f} | "
            f"flip={m['sign_flip_mean']:.4f}"
        )

    keys = [
        "diff_l2_mean",
        "diff_l2_max",
        "diff_abs_mean",
        "accel_l2_mean",
        "accel_l2_max",
        "accel_abs_mean",
        "jerk_l2_mean",
        "jerk_l2_max",
        "jerk_abs_mean",
        "boundary_jump_l2_mean",
        "boundary_jump_l2_max",
        "sign_flip_mean",
    ]

    summary = {k: mean_scalar(metrics, k) for k in keys}

    print("\n[MEAN]")
    for k in keys:
        print(f"{k:28s} = {summary[k]:.8f}")

    jerk_dim = mean_array(metrics, "jerk_abs_per_dim")
    accel_dim = mean_array(metrics, "accel_abs_per_dim")
    diff_dim = mean_array(metrics, "diff_abs_per_dim")
    flip_dim = mean_array(metrics, "sign_flip_per_dim")

    if jerk_dim is not None:
        order = np.argsort(-jerk_dim)
        print("\n[Top dims by executed jerk_abs]")
        for rank, d in enumerate(order[:5], 1):
            print(
                f"rank {rank}: dim={d}, "
                f"jerk_abs={jerk_dim[d]:.8f}, "
                f"accel_abs={accel_dim[d]:.8f}, "
                f"diff_abs={diff_dim[d]:.8f}, "
                f"sign_flip={flip_dim[d]:.4f}"
            )

    return metrics, summary


def print_comparison(base_name: str, test_name: str, base_summary: dict, test_summary: dict):
    print("\n" + "=" * 100)
    print(f"[Relative change] {test_name} - {base_name}")
    print("=" * 100)

    keys = [
        "diff_l2_mean",
        "accel_l2_mean",
        "jerk_l2_mean",
        "boundary_jump_l2_mean",
        "sign_flip_mean",
        "diff_l2_max",
        "accel_l2_max",
        "jerk_l2_max",
        "boundary_jump_l2_max",
    ]

    for k in keys:
        b = base_summary[k]
        t = test_summary[k]
        rel = (t - b) / (b + 1e-12) * 100.0
        print(f"{k:28s}: {b:.8f} -> {t:.8f} | {rel:+.4f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="baseline action log file or directory")
    parser.add_argument("--test_path", type=str, required=True, help="test action log file or directory")
    parser.add_argument("--base_name", type=str, default="baseline")
    parser.add_argument("--test_name", type=str, default="test")
    parser.add_argument("--n_action_steps", type=int, default=80)
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    base_files = find_episode_files(args.base_path)
    test_files = find_episode_files(args.test_path)

    n_pair = min(len(base_files), len(test_files))
    if args.max_episodes is not None:
        n_pair = min(n_pair, args.max_episodes)

    base_files = base_files[:n_pair]
    test_files = test_files[:n_pair]

    print("=" * 100)
    print("[Compare rollout executed-action smoothness]")
    print(f"base_path       = {args.base_path}")
    print(f"test_path       = {args.test_path}")
    print(f"base_name       = {args.base_name}")
    print(f"test_name       = {args.test_name}")
    print(f"n_action_steps  = {args.n_action_steps}")
    print(f"episodes used   = {n_pair}")
    print("=" * 100)

    _, base_summary = summarize_group(args.base_name, base_files, args.n_action_steps, n_pair)
    _, test_summary = summarize_group(args.test_name, test_files, args.n_action_steps, n_pair)

    print_comparison(args.base_name, args.test_name, base_summary, test_summary)

    print("\n" + "=" * 100)
    print("[Metric guide]")
    print("diff      : executed action step-to-step change")
    print("accel     : executed action 2nd difference. 예전 코드의 jerk_l2가 사실상 이 값이었음")
    print("jerk      : executed action 3rd difference")
    print("boundary  : chunk boundary에서 action[b] - action[b-1] 크기")
    print("sign_flip : velocity 방향 전환 비율")
    print("=" * 100)


if __name__ == "__main__":
    main()
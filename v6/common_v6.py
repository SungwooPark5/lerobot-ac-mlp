"""common_v6.py — shared utilities for all v6 experiment notebooks.

Usage:
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    import common_v6 as v6
"""

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

# ── Project paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path.home() / "lerobot_project" / "lerobot-ac-mlp"
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "v6_seed0"
SRC_DIR = PROJECT_ROOT / "src"

# ── Dataset / env ──────────────────────────────────────────────────────────────

DATASET = "lerobot/aloha_sim_transfer_cube_human"
ENV_TYPE = "aloha"
ENV_TASK = "AlohaTransferCube-v0"

# ── Training hyper-params (shared defaults) ────────────────────────────────────

STEPS = 200_000
BATCH_SIZE = 8
EVAL_FREQ = 10_000
EVAL_START = 60_000
TRAIN_EVAL_N = 50
SAVE_FREQ = 10_000
SEED = 0

# ── Eval hyper-params ─────────────────────────────────────────────────────────

EVAL_N_EPISODES = 500
EVAL_BATCH_SIZE = 50

# ── Model configs ─────────────────────────────────────────────────────────────
# Each entry: tag → (policy_type, lr, chunk_size, extra_flags: list[str], use_chunk_pairs)
# extra_flags are appended verbatim to the training command.

MODEL_CONFIGS: dict[str, tuple[str, float, int, list[str], bool]] = {
    # ── Baselines ────────────────────────────────────────────────────────────
    "B1": ("act",   1e-4, 100, [], False),
    "B2": ("acm",   1e-4, 100, [], False),
    "B3": ("acm2",  1e-4, 100, [], False),
    "B4": ("acm3",  1e-4, 100, [], False),
    # ── Control ──────────────────────────────────────────────────────────────
    "N1": ("act_icpe", 1e-4, 100, [], False),
    # ── Ablations ────────────────────────────────────────────────────────────
    "A1": ("acm3_icpe", 1e-4, 100, ["--policy.icpe_mode=sincos"], False),
    "A2": ("acm3_icpe", 1e-4, 100, ["--policy.icpe_mode=linear"], False),
    "A3": ("acm3_icpe", 1e-4, 100, [], False),
    "A4": ("acm3_sscp", 1e-4, 100, [], False),
    # ── CC variants ──────────────────────────────────────────────────────────
    "A4cc": ("acm3_sscp",       1e-4, 100, ["--policy.sscp_p_carry=0.5"], True),
    "P1cc": ("acm3_icpe_tsscp", 1e-4, 100, ["--policy.sscp_p_carry=0.5"], True),
    # ── Full model ───────────────────────────────────────────────────────────
    "P1": ("acm3_icpe_tsscp", 1e-4, 100, [], False),
    # ── K-scaling: ACT ───────────────────────────────────────────────────────
    "K1": ("act",  1e-4,  50, [], False),
    "K2": ("act",  1e-4, 200, [], False),
    # ── K-scaling: ACM3 ──────────────────────────────────────────────────────
    "K3": ("acm3", 1e-4,  50, [], False),
    "K4": ("acm3", 1e-4, 200, [], False),
    # ── K-scaling: Full model ────────────────────────────────────────────────
    "K5": ("acm3_icpe_tsscp", 1e-4,  50, [], False),
    "K6": ("acm3_icpe_tsscp", 1e-4, 200, [], False),
}

# Human-readable labels for tables/figures
MODEL_LABELS: dict[str, str] = {
    "B1": "ACT (k=100)",
    "B2": "ACM (k=100)",
    "B3": "ACM2 (k=100)",
    "B4": "ACM3 (k=100)",
    "N1": "ACT+ICPE (k=100)",
    "A1": "ACM3+ICPE[sincos] (k=100)",
    "A2": "ACM3+ICPE[linear] (k=100)",
    "A3": "ACM3+ICPE (k=100)",
    "A4": "ACM3+SSCP[inf] (k=100)",
    "A4cc": "ACM3+SSCP[CC] (k=100)",
    "P1":   "ACM3+ICPE+SSCP (k=100)",
    "P1cc": "ACM3+ICPE+SSCP[CC] (k=100)",
    "K1": "ACT (k=50)",
    "K2": "ACT (k=200)",
    "K3": "ACM3 (k=50)",
    "K4": "ACM3 (k=200)",
    "K5": "Full (k=50)",
    "K6": "Full (k=200)",
}

# ── Output directory helpers ──────────────────────────────────────────────────

def get_output_dir(tag: str) -> Path:
    policy_type, _, chunk_size, extra_flags, use_cp = MODEL_CONFIGS[tag]
    suffix = "cc" if use_cp else ""
    name = f"{policy_type}_k{chunk_size}{suffix}"
    return OUTPUT_BASE / tag / name


def get_last_checkpoint(tag: str) -> Path | None:
    """Return the path to the latest checkpoint directory for `tag`, or None."""
    out = get_output_dir(tag)
    ckpt_root = out / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    checkpoints = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )
    return checkpoints[-1] if checkpoints else None


# ── Training helpers ──────────────────────────────────────────────────────────

def make_train_cmd(tag: str, gpu_id: int) -> str:
    policy_type, lr, chunk_size, extra_flags, use_cp = MODEL_CONFIGS[tag]
    out_dir = get_output_dir(tag)
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python -m lerobot.scripts.lerobot_train",
        f"--policy.type={policy_type}",
        f"--policy.chunk_size={chunk_size}",
        f"--dataset.repo_id={DATASET}",
        f"--env.type={ENV_TYPE}",
        f"--env.task={ENV_TASK}",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        f"--eval_freq={EVAL_FREQ}",
        f"--eval_start_step={EVAL_START}",
        f"--eval.n_episodes={TRAIN_EVAL_N}",
        f"--save_freq={SAVE_FREQ}",
        f"--seed={SEED}",
        f"--output_dir={out_dir}",
    ]
    if use_cp:
        parts.append("--use_chunk_pairs=true")
    parts.extend(extra_flags)
    return " ".join(parts)


def launch_training(tags: list[str], gpu_offset: int = 0) -> dict[str, subprocess.Popen]:
    """Launch training for each tag on GPU gpu_offset+i. Returns {tag: Popen}."""
    procs: dict[str, subprocess.Popen] = {}
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    for i, tag in enumerate(tags):
        gpu_id = gpu_offset + i
        cmd = make_train_cmd(tag, gpu_id)
        out_dir = get_output_dir(tag)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = out_dir / "train.log"
        print(f"[{tag}] GPU={gpu_id}  → {out_dir.name}")
        print(f"       cmd: {cmd[:120]}...")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, shell=True, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        procs[tag] = proc
    return procs


# ── Training status ────────────────────────────────────────────────────────────

def get_training_status(tag: str) -> dict[str, Any]:
    """Parse train.log to extract current step and best eval SR."""
    out_dir = get_output_dir(tag)
    log_file = out_dir / "train.log"
    status: dict[str, Any] = {
        "tag": tag,
        "step": 0,
        "best_sr": None,
        "done": False,
        "log_exists": log_file.exists(),
    }
    if not log_file.exists():
        return status

    last_step = 0
    best_sr = None
    with open(log_file) as f:
        for line in f:
            if "step:" in line or "global_step" in line:
                for tok in line.split():
                    if tok.isdigit():
                        s = int(tok)
                        if s > last_step:
                            last_step = s
            if "eval/avg_sum_rewards" in line or "success_rate" in line:
                for tok in line.split():
                    try:
                        v = float(tok)
                        if 0 <= v <= 1:
                            if best_sr is None or v > best_sr:
                                best_sr = v
                    except ValueError:
                        pass
    status["step"] = last_step
    status["best_sr"] = best_sr
    status["done"] = last_step >= STEPS
    return status


def print_training_status(tags: list[str]) -> None:
    """Print a status table for the given tags."""
    print(f"{'TAG':<8} {'STEP':>8} {'BEST_SR':>8} {'DONE':>6}  DIR")
    print("-" * 72)
    for tag in tags:
        s = get_training_status(tag)
        sr_str = f"{s['best_sr']:.3f}" if s["best_sr"] is not None else "  ─"
        done_str = "YES" if s["done"] else "no"
        out = get_output_dir(tag)
        print(f"{tag:<8} {s['step']:>8,} {sr_str:>8} {done_str:>6}  {out.name}")


# ── Eval helpers ──────────────────────────────────────────────────────────────

def make_eval_cmd(tag: str, gpu_id: int, n_episodes: int | None = None,
                  checkpoint: str | Path | None = None) -> str:
    """Build an eval command for `tag`.

    checkpoint: path to checkpoint dir, or 'last' to auto-find. Defaults to last.
    """
    if checkpoint is None or checkpoint == "last":
        ckpt = get_last_checkpoint(tag)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found for tag={tag}")
    else:
        ckpt = Path(checkpoint)

    n = n_episodes or EVAL_N_EPISODES
    out_dir = get_output_dir(tag) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_type = MODEL_CONFIGS[tag][0]
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python -m lerobot.scripts.eval",
        f"--policy.path={ckpt}",
        f"--env.type={ENV_TYPE}",
        f"--env.task={ENV_TASK}",
        f"--eval.n_episodes={n}",
        f"--eval.batch_size={EVAL_BATCH_SIZE}",
        f"--output_dir={out_dir}",
    ]
    return " ".join(parts)


def launch_eval(jobs: list[tuple[str, int, int | None]], gpu_offset: int = 0) -> dict[str, subprocess.Popen]:
    """Launch eval jobs. jobs = [(tag, gpu_local_id, n_episodes), ...].

    gpu_local_id is relative to gpu_offset.
    """
    procs: dict[str, subprocess.Popen] = {}
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR),
           "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    for tag, gpu_local, n_ep in jobs:
        gpu_id = gpu_offset + gpu_local
        cmd = make_eval_cmd(tag, gpu_id, n_episodes=n_ep)
        out_dir = get_output_dir(tag) / "eval"
        log_file = out_dir / "eval.log"
        print(f"[{tag}] GPU={gpu_id}  n={n_ep or EVAL_N_EPISODES}")
        print(f"       cmd: {cmd[:120]}...")
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, shell=True, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )
        procs[tag] = proc
    return procs


# ── Eval status ───────────────────────────────────────────────────────────────

def get_eval_status(tag: str) -> dict[str, Any]:
    """Parse eval output directory for results."""
    eval_dir = get_output_dir(tag) / "eval"
    log_file = eval_dir / "eval.log"
    status: dict[str, Any] = {
        "tag": tag,
        "sr": None,
        "n_done": 0,
        "done": False,
        "log_exists": log_file.exists(),
    }
    if not log_file.exists():
        return status

    n_done = 0
    sr = None
    with open(log_file) as f:
        for line in f:
            if "success" in line.lower():
                for tok in line.split():
                    try:
                        v = float(tok)
                        if 0 <= v <= 1:
                            sr = v
                    except ValueError:
                        pass
            if "episode" in line.lower():
                for tok in line.split():
                    if tok.isdigit():
                        n_done = max(n_done, int(tok))
    status["sr"] = sr
    status["n_done"] = n_done
    status["done"] = n_done >= EVAL_N_EPISODES
    return status


def print_eval_status(tags: list[str]) -> None:
    print(f"{'TAG':<8} {'N_DONE':>8} {'SR':>8} {'DONE':>6}")
    print("-" * 40)
    for tag in tags:
        s = get_eval_status(tag)
        sr_str = f"{s['sr']:.3f}" if s["sr"] is not None else "  ─"
        done_str = "YES" if s["done"] else "no"
        print(f"{tag:<8} {s['n_done']:>8} {sr_str:>8} {done_str:>6}")


# ── Metrics helpers ───────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def compute_jitter_profile(
    action_trajectory: list[list[float]],
) -> dict[str, float]:
    """Compute jitter metrics from a list of action vectors.

    Returns mean L2 diff, max L2 diff, and spectral power ratio (high/total).
    """
    import numpy as np

    traj = np.array(action_trajectory)  # (T, D)
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    mean_jitter = float(diffs.mean())
    max_jitter = float(diffs.max())

    # Spectral ratio: power above Nyquist/2 vs total
    fft = np.fft.rfft(traj, axis=0)
    power = np.abs(fft) ** 2
    n_freq = power.shape[0]
    mid = n_freq // 2
    high_power = float(power[mid:].sum())
    total_power = float(power.sum()) + 1e-12
    spectral_ratio = high_power / total_power

    return {
        "mean_jitter": mean_jitter,
        "max_jitter": max_jitter,
        "spectral_ratio": spectral_ratio,
    }


def save_metrics(tag: str, metrics: dict[str, Any]) -> None:
    out_dir = get_output_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(tag: str) -> dict[str, Any] | None:
    path = get_output_dir(tag) / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_metrics(tags: list[str]) -> dict[str, dict[str, Any]]:
    return {t: m for t in tags if (m := load_metrics(t)) is not None}


def build_metrics_from_eval_info(tag: str, eval_info_path: str | Path | None = None) -> dict[str, Any]:
    """Build a metrics dict from an eval output JSON if available."""
    if eval_info_path is None:
        eval_info_path = get_output_dir(tag) / "eval" / "eval_info.json"
    path = Path(eval_info_path)
    if not path.exists():
        return {}
    with open(path) as f:
        info = json.load(f)
    n = info.get("n_episodes", EVAL_N_EPISODES)
    k = int(round(info.get("avg_sum_rewards", 0) * n))
    lo, hi = wilson_ci(k, n)
    return {
        "tag": tag,
        "label": MODEL_LABELS.get(tag, tag),
        "sr": info.get("avg_sum_rewards"),
        "sr_ci_lo": lo,
        "sr_ci_hi": hi,
        "n_episodes": n,
    }


# ── Misc ──────────────────────────────────────────────────────────────────────

def wait_for_procs(procs: dict[str, subprocess.Popen], poll_interval: int = 60) -> None:
    """Block until all processes finish, printing status every poll_interval s."""
    remaining = dict(procs)
    while remaining:
        done = []
        for tag, proc in remaining.items():
            rc = proc.poll()
            if rc is not None:
                print(f"[{tag}] finished (rc={rc})")
                done.append(tag)
        for tag in done:
            del remaining[tag]
        if remaining:
            print(f"Still running: {list(remaining.keys())}  — sleeping {poll_interval}s")
            time.sleep(poll_interval)
    print("All processes done.")

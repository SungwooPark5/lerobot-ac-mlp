import os
import json
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# Configuration
# ============================================================

BASE = "/home1/eunji24/lerobot_project/outputs/v23/train/transfer/valtest_1e-5"

# Example:
#   STEP=videos_step_100000 python measure_smoothness_final.py
STEP = os.environ.get("STEP", "videos_step_200000")

POLICIES = [
    "bimamba_only",
    "overlap_only",
]

# Aggregate over every training seed (5 seeds x 50 episodes per step).
SEEDS = [
    int(s)
    for s in os.environ.get("SEEDS", "0,1,2,3,4").split(",")
    if s.strip() != ""
]

# ALOHA control frequency.
CONTROL_HZ = 50.0

# Chunk execution settings.
CHUNK_SIZE = 100
OVERLAP_SIZE = 10

# Boundary samples are defined as jerk samples located
# within +/- 2 control steps of a chunk transition.
BOUNDARY_RADIUS = 2.0

# Standard policies hard-switch every 100 actions.
# Ours actually introduces a new chunk every 90 actions (hop = chunk size - overlap = 100 - 10),
# but for an apples-to-apples boundary comparison we evaluate boundary jerk at the SAME locations
# (every 100 steps) as the other models.
BOUNDARY_STRIDES = {
    "bimamba_only": CHUNK_SIZE,
    "overlap_only": CHUNK_SIZE,
}


def runs_for(policy):
    """
    Match seed directories named:
        {BASE}/{policy}/seed{S}
    for every seed in SEEDS.
    """
    return [
        Path(BASE) / policy / f"seed{seed}"
        for seed in SEEDS
    ]


ROOTS = {
    policy: [
        d / "eval" / STEP
        for d in runs_for(policy)
    ]
    for policy in POLICIES
}


STEP_NUMBER = STEP.replace("videos_step_", "")

OUT_SUMMARY_CSV = (
    f"smoothness_ablation_1e-5_{STEP_NUMBER}_summary.csv"
)

OUT_PER_TRAJ_CSV = (
    f"smoothness_ablation_1e-5_{STEP_NUMBER}_per_trajectory.csv"
)


# ============================================================
# File loading
# ============================================================

def find_action_logs(root):
    """
    Find all action_logs directories recursively.

    root can be:
        - one path
        - a list of paths
    """
    roots = root if isinstance(root, (list, tuple)) else [root]

    directories = []

    for current_root in roots:
        current_root = Path(current_root)

        if not current_root.exists():
            continue

        directories.extend(current_root.rglob("action_logs"))

    return sorted(set(directories))


def load_action_file(path):
    """
    Load actions from a supported file format.
    """
    path = Path(path)

    candidate_keys = [
        "actions",
        "action",
        "executed_actions",
        "pred_actions",
        "raw_actions",
    ]

    if path.suffix in {".pt", ".pth"}:
        import torch

        data = torch.load(path, map_location="cpu")

        if isinstance(data, dict):
            for key in candidate_keys:
                if key in data:
                    value = data[key]

                    if hasattr(value, "detach"):
                        return value.detach().cpu().numpy()

                    return np.asarray(value)

        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()

        return np.asarray(data)

    if path.suffix == ".npy":
        return np.load(path)

    if path.suffix == ".npz":
        data = np.load(path)

        for key in candidate_keys:
            if key in data:
                return data[key]

        if len(data.files) == 0:
            return None

        return data[data.files[0]]

    if path.suffix == ".csv":
        return pd.read_csv(path).values

    if path.suffix == ".json":
        with open(path, "r") as file:
            data = json.load(file)

        if isinstance(data, dict):
            for key in candidate_keys:
                if key in data:
                    return np.asarray(data[key])

        return np.asarray(data)

    return None


def collect_action_arrays(log_dir):
    """
    Collect executed action trajectories.

    Supported shapes:
        [T, action_dim]
        [episode, T, action_dim]
    """
    log_dir = Path(log_dir)
    arrays = []

    supported_suffixes = {
        ".pt",
        ".pth",
        ".npy",
        ".npz",
        ".csv",
        ".json",
    }

    for path in sorted(log_dir.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in supported_suffixes:
            continue

        try:
            array = load_action_file(path)
        except Exception as error:
            print(f"[WARN] failed to load {path}: {error}")
            continue

        if array is None:
            continue

        try:
            array = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError):
            continue

        # One trajectory: [T, D]
        if (
            array.ndim == 2
            and array.shape[0] >= 4
            and array.shape[-1] <= 64
        ):
            arrays.append(array)

        # Multiple trajectories: [N, T, D]
        elif (
            array.ndim == 3
            and array.shape[1] >= 4
            and array.shape[-1] <= 64
        ):
            for episode_index in range(array.shape[0]):
                arrays.append(array[episode_index])

    return arrays


# ============================================================
# Utilities
# ============================================================

def safe_norm(x, axis=-1):
    return np.linalg.norm(x, axis=axis)


def safe_rms(x):
    x = np.asarray(x, dtype=np.float64)

    if x.size == 0:
        return np.nan

    return np.sqrt(np.mean(x ** 2))


# ============================================================
# Finite-difference jerk
# ============================================================

def third_forward_difference(actions):
    """
    Third-order forward finite difference:

        Delta^3 a_t
        = a_{t+3} - 3a_{t+2} + 3a_{t+1} - a_t

    Input:
        actions: [T, A]

    Output:
        jerk: [T - 3, A]
    """
    return np.diff(actions, n=3, axis=0)


def jerk_sample_centers(num_action_steps):
    """
    Delta^3 a_t uses actions at:
        t, t+1, t+2, t+3

    We assign this jerk sample to the temporal center:
        t + 1.5
    """
    num_jerk_samples = num_action_steps - 3

    if num_jerk_samples <= 0:
        return np.asarray([], dtype=np.float64)

    return (
        np.arange(num_jerk_samples, dtype=np.float64)
        + 1.5
    )


def chunk_transition_steps(num_action_steps, stride):
    """
    Return the chunk-transition steps.

    The first chunk starts at 0, so the first transition
    occurs at 'stride'.
    """
    return np.arange(
        stride,
        num_action_steps,
        stride,
        dtype=np.float64,
    )


def nearest_transition_offsets(sample_centers, transitions):
    """
    Signed distance from each jerk sample center to its nearest
    chunk transition.

    Negative: before the transition
    Positive: after the transition
    """
    if len(sample_centers) == 0 or len(transitions) == 0:
        return np.full(
            len(sample_centers),
            np.nan,
            dtype=np.float64,
        )

    distances = (
        sample_centers[:, None]
        - transitions[None, :]
    )

    nearest_indices = np.argmin(
        np.abs(distances),
        axis=1,
    )

    return distances[
        np.arange(len(sample_centers)),
        nearest_indices,
    ]


def boundary_and_interior_jerk(
    actions,
    boundary_stride,
    radius=BOUNDARY_RADIUS,
    eps=1e-12,
):
    """
    Compute boundary and interior jerk statistics.

    Boundary:
        jerk samples whose temporal centers lie within
        +/- radius control steps of a chunk transition.

    Interior:
        all remaining valid jerk samples.
    """
    actions = np.asarray(actions, dtype=np.float64)

    empty_result = {
        "boundary_jerk_rms": np.nan,
        "interior_jerk_rms": np.nan,
        "boundary_interior_difference": np.nan,
        "boundary_interior_ratio": np.nan,
        "boundary_jerk_mean": np.nan,
        "interior_jerk_mean": np.nan,
        "n_boundaries": 0,
        "n_boundary_samples": 0,
        "n_interior_samples": 0,
    }

    if actions.ndim != 2 or actions.shape[0] < 4:
        return empty_result

    jerk = third_forward_difference(actions)

    # One scalar jerk magnitude per time step.
    jerk_l2 = safe_norm(jerk, axis=-1)

    centers = jerk_sample_centers(
        num_action_steps=actions.shape[0]
    )

    transitions = chunk_transition_steps(
        num_action_steps=actions.shape[0],
        stride=boundary_stride,
    )

    if len(transitions) == 0:
        return empty_result

    offsets = nearest_transition_offsets(
        sample_centers=centers,
        transitions=transitions,
    )

    boundary_mask = np.abs(offsets) <= radius
    interior_mask = ~boundary_mask

    boundary_values = jerk_l2[boundary_mask]
    interior_values = jerk_l2[interior_mask]

    if (
        len(boundary_values) == 0
        or len(interior_values) == 0
    ):
        result = empty_result.copy()
        result.update({
            "n_boundaries": len(transitions),
            "n_boundary_samples": len(boundary_values),
            "n_interior_samples": len(interior_values),
        })
        return result

    boundary_rms = safe_rms(boundary_values)
    interior_rms = safe_rms(interior_values)

    return {
        "boundary_jerk_rms": boundary_rms,
        "interior_jerk_rms": interior_rms,

        # Difference between RMS values.
        "boundary_interior_difference":
            boundary_rms - interior_rms,

        # Ratio close to 1 means the boundary is no rougher
        # than the chunk interior.
        "boundary_interior_ratio":
            boundary_rms / (interior_rms + eps),

        "boundary_jerk_mean":
            np.mean(boundary_values),

        "interior_jerk_mean":
            np.mean(interior_values),

        "n_boundaries": len(transitions),
        "n_boundary_samples": len(boundary_values),
        "n_interior_samples": len(interior_values),
    }


# ============================================================
# SPARC from scalar action-speed profile
# ============================================================

def sparc_1d(
    signal,
    fs,
    padlevel=4,
    fc=10.0,
    amp_th=0.05,
    eps=1e-12,
):
    """
    Spectral Arc Length.

    Closer to zero indicates smoother motion.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError(
            f"SPARC signal must be 1-D, got {signal.shape}"
        )

    n = len(signal)

    if n < 8:
        return np.nan

    if fs <= 0:
        raise ValueError(
            f"Sampling frequency must be positive, got {fs}"
        )

    signal = signal - np.mean(signal)

    if np.max(np.abs(signal)) < eps:
        return 0.0

    base_nfft = 2 ** int(np.ceil(np.log2(n)))
    nfft = base_nfft * (2 ** padlevel)

    frequencies = np.fft.rfftfreq(
        nfft,
        d=1.0 / fs,
    )

    magnitude = np.abs(
        np.fft.rfft(signal, n=nfft)
    )

    max_magnitude = np.max(magnitude)

    if max_magnitude < eps:
        return 0.0

    magnitude = magnitude / (
        max_magnitude + eps
    )

    # Limit the spectrum to at most 10 Hz.
    effective_cutoff = min(fc, fs / 2.0)

    frequency_mask = (
        frequencies <= effective_cutoff
    )

    frequencies = frequencies[frequency_mask]
    magnitude = magnitude[frequency_mask]

    above_threshold = np.where(
        magnitude >= amp_th
    )[0]

    if len(above_threshold) < 2:
        return 0.0

    # Retain the spectrum up to the final frequency bin whose
    # normalized magnitude exceeds the threshold.
    last_index = above_threshold[-1] + 1

    frequencies = frequencies[:last_index]
    magnitude = magnitude[:last_index]

    if len(frequencies) < 2:
        return 0.0

    normalized_frequency = frequencies / (
        frequencies[-1] + eps
    )

    delta_frequency = np.diff(
        normalized_frequency
    )

    delta_magnitude = np.diff(
        magnitude
    )

    spectral_arc_length = -np.sum(
        np.sqrt(
            delta_frequency ** 2
            + delta_magnitude ** 2
        )
    )

    return float(spectral_arc_length)


def sparc_from_speed_profile(
    actions,
    fs=CONTROL_HZ,
    fc=10.0,
    amp_th=0.05,
):
    """
    Compute SPARC from:

        s_t = ||a_{t+1} - a_t||_2
    """
    actions = np.asarray(
        actions,
        dtype=np.float64,
    )

    if actions.ndim != 2 or actions.shape[0] < 9:
        return np.nan

    action_difference = np.diff(
        actions,
        axis=0,
    )

    speed_profile = safe_norm(
        action_difference,
        axis=-1,
    )

    return sparc_1d(
        speed_profile,
        fs=fs,
        padlevel=4,
        fc=fc,
        amp_th=amp_th,
    )


# ============================================================
# Complementary metrics
# ============================================================

def log_dimensionless_jerk_style(
    actions,
    eps=1e-12,
):
    """
    LDJ-style cost retained from the original implementation.

    ldj:
        larger / less negative is smoother-like

    ldj_cost:
        -ldj, so smaller is smoother-like
    """
    actions = np.asarray(
        actions,
        dtype=np.float64,
    )

    num_steps = actions.shape[0]

    if num_steps < 4:
        return np.nan

    velocity = np.diff(
        actions,
        axis=0,
    )

    speed = safe_norm(
        velocity,
        axis=-1,
    )

    amplitude = np.sum(speed) + eps

    jerk = third_forward_difference(
        actions
    )

    jerk_squared_sum = np.sum(
        jerk ** 2
    )

    duration = float(num_steps - 1)

    value = (
        duration ** 5
        / (amplitude ** 2 + eps)
    ) * jerk_squared_sum + eps

    return -np.log(value)


def sign_flip_rate(actions):
    """
    Frequency of sign changes in consecutive first-order
    action differences, averaged over time and action dimensions.
    """
    action_difference = np.diff(
        actions,
        axis=0,
    )

    if len(action_difference) < 2:
        return np.nan

    flips = (
        action_difference[1:]
        * action_difference[:-1]
    ) < 0

    return np.mean(flips)


# ============================================================
# Per-trajectory metrics
# ============================================================

def metrics_for_trajectory(
    actions,
    boundary_stride,
):
    """
    Calculate all metrics described in the paper.
    """
    actions = np.asarray(
        actions,
        dtype=np.float64,
    )

    if (
        actions.ndim != 2
        or actions.shape[0] < 4
    ):
        return None

    first_difference = np.diff(
        actions,
        n=1,
        axis=0,
    )

    second_difference = np.diff(
        actions,
        n=2,
        axis=0,
    )

    jerk = third_forward_difference(
        actions
    )

    jerk_l2 = safe_norm(
        jerk,
        axis=-1,
    )

    ldj = log_dimensionless_jerk_style(
        actions
    )

    metrics = {
        "T": actions.shape[0],
        "D": actions.shape[1],

        # Optional basic diagnostics.
        "diff_abs_mean":
            np.mean(np.abs(first_difference)),

        "accel_abs_mean":
            np.mean(np.abs(second_difference)),

        # Primary overall jerk metric:
        # RMS over time and action dimensions.
        "jerk_rms":
            np.sqrt(np.mean(jerk ** 2)),

        # Additional vector-magnitude diagnostics.
        "jerk_l2_mean":
            np.mean(jerk_l2),

        "jerk_l2_rms":
            safe_rms(jerk_l2),

        "jerk_l2_p95":
            np.percentile(jerk_l2, 95),

        # Global smoothness.
        "sparc":
            sparc_from_speed_profile(
                actions,
                fs=CONTROL_HZ,
                fc=10.0,
                amp_th=0.05,
            ),

        # Complementary global jitter metrics.
        "ldj":
            ldj,

        "ldj_cost":
            -ldj if np.isfinite(ldj) else np.nan,

        "sign_flip_rate":
            sign_flip_rate(actions),
    }

    metrics.update(
        boundary_and_interior_jerk(
            actions,
            boundary_stride=boundary_stride,
            radius=BOUNDARY_RADIUS,
        )
    )

    return metrics


# ============================================================
# Model summary
# ============================================================

def summarize_model(name, root):
    """
    Calculate per-trajectory and aggregate metrics for one model.
    """
    if name not in BOUNDARY_STRIDES:
        raise KeyError(
            f"No boundary stride configured for {name}"
        )

    boundary_stride = BOUNDARY_STRIDES[name]

    log_directories = find_action_logs(root)

    if not log_directories:
        print(
            f"[WARN] no action_logs found: {name} {root}"
        )
        return None, None

    all_arrays = []

    for directory in log_directories:
        all_arrays.extend(
            collect_action_arrays(directory)
        )

    if not all_arrays:
        print(
            f"[WARN] no action arrays found: {name}"
        )
        return None, None

    rows = []

    for trajectory_id, array in enumerate(all_arrays):
        metrics = metrics_for_trajectory(
            array,
            boundary_stride=boundary_stride,
        )

        if metrics is None:
            continue

        metrics["model"] = name
        metrics["trajectory_id"] = trajectory_id

        rows.append(metrics)

    if not rows:
        print(
            f"[WARN] no valid trajectories: {name}"
        )
        return None, None

    trajectory_df = pd.DataFrame(rows)

    summary = {
        "model": name,
        "n_traj": len(trajectory_df),
        "boundary_stride": boundary_stride,
        "T_mean": trajectory_df["T"].mean(),
        "D_mean": trajectory_df["D"].mean(),
    }

    excluded = {
        "model",
        "trajectory_id",
        "T",
        "D",
    }

    metric_columns = [
        column
        for column in trajectory_df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(
            trajectory_df[column]
        )
    ]

    for column in metric_columns:
        summary[f"{column}_mean"] = (
            trajectory_df[column].mean()
        )

        summary[f"{column}_std"] = (
            trajectory_df[column].std()
        )

    return summary, trajectory_df


# ============================================================
# Main
# ============================================================

def main():
    summaries = []
    trajectory_frames = []

    print("=======================================")
    print("ALOHA smoothness analysis")
    print("=======================================")
    print(f"Checkpoint: {STEP}")
    print(f"Policies: {', '.join(POLICIES)}")
    print(f"Seeds (aggregated): {SEEDS}")
    print(f"Control frequency: {CONTROL_HZ} Hz")
    print(f"Boundary radius: +/- {BOUNDARY_RADIUS}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Overlap: {OVERLAP_SIZE}")
    print("Boundary strides:")

    for policy, stride in BOUNDARY_STRIDES.items():
        print(f"  {policy}: {stride}")

    for name, root in ROOTS.items():
        print(f"\n===== {name} =====")

        summary, trajectory_df = summarize_model(
            name,
            root,
        )

        if summary is None:
            continue

        summaries.append(summary)
        trajectory_frames.append(trajectory_df)

        print(
            pd.Series(summary).to_string()
        )

    if not summaries:
        raise RuntimeError(
            "No valid trajectories were found."
        )

    summary_df = pd.DataFrame(summaries)

    per_trajectory_df = pd.concat(
        trajectory_frames,
        ignore_index=True,
    )

    summary_df.to_csv(
        OUT_SUMMARY_CSV,
        index=False,
    )

    per_trajectory_df.to_csv(
        OUT_PER_TRAJ_CSV,
        index=False,
    )

    compact_columns = [
        "model",
        "n_traj",
        "boundary_stride",

        # Overall finite-difference jerk.
        "jerk_rms_mean",

        # Boundary/interior analysis.
        "boundary_jerk_rms_mean",
        "interior_jerk_rms_mean",
        "boundary_interior_difference_mean",
        "boundary_interior_ratio_mean",

        # Global smoothness and jitter.
        "sparc_mean",
        "ldj_cost_mean",
        "sign_flip_rate_mean",
    ]

    compact_columns = [
        column
        for column in compact_columns
        if column in summary_df.columns
    ]

    print("\n=======================================")
    print("Compact summary")
    print("=======================================")

    print(
        summary_df[
            compact_columns
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {OUT_SUMMARY_CSV}")
    print(f"  {OUT_PER_TRAJ_CSV}")

    print("\nMetric directions:")
    print("  Jerk RMS: lower is smoother")
    print("  Boundary jerk RMS: lower is smoother")
    print("  Interior jerk RMS: lower is smoother")
    print("  B/I ratio: closer to 1 is better")
    print("  SPARC: closer to 0 is smoother")
    print("  LDJ cost: lower is smoother-like")
    print("  Sign-flip rate: lower is smoother")


if __name__ == "__main__":
    main()
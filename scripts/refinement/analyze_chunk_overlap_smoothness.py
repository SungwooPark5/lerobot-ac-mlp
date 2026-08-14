#!/usr/bin/env python

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def get_scalar(x):
    if torch.is_tensor(x):
        if x.numel() == 1:
            return int(x.item())
        return int(x.flatten()[0].item())
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


def l2_mean(x):
    # x: (..., action_dim)
    return torch.linalg.norm(x, dim=-1).mean().item()


def abs_mean(x):
    return x.abs().mean().item()


def per_dim_abs_mean(x):
    # x: (..., action_dim)
    dims = tuple(range(x.ndim - 1))
    return x.abs().mean(dim=dims).detach().cpu().numpy()


def per_dim_abs_max(x):
    dims = tuple(range(x.ndim - 1))
    return x.abs().amax(dim=dims).detach().cpu().numpy()


def sign_flip_rate(actions, eps=1e-6):
    """
    actions: (T, A) or (B, T, A)
    velocity sign flip rate per action dimension.
    """
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)

    diff = actions[:, 1:] - actions[:, :-1]  # (B, T-1, A)

    if diff.shape[1] < 2:
        return np.zeros(diff.shape[-1], dtype=np.float32), 0.0

    d0 = diff[:, :-1]
    d1 = diff[:, 1:]

    valid = (d0.abs() > eps) & (d1.abs() > eps)
    flips = (d0 * d1 < 0) & valid

    denom = valid.sum(dim=(0, 1)).clamp(min=1)
    per_dim = flips.sum(dim=(0, 1)).float() / denom.float()
    return per_dim.detach().cpu().numpy(), per_dim.mean().item()


def chunk_metrics(actions):
    """
    actions: (B, T, A) or (T, A)

    chunk_:
      diff  = a[t+1] - a[t]
      accel = a[t+2] - 2a[t+1] + a[t]
      jerk  = a[t+3] - 3a[t+2] + 3a[t+1] - a[t]
    """
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)

    out = {}

    diff = actions[:, 1:] - actions[:, :-1]
    accel = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    jerk = (
        actions[:, 3:]
        - 3.0 * actions[:, 2:-1]
        + 3.0 * actions[:, 1:-2]
        - actions[:, :-3]
    )

    out["chunk_diff_l2_mean"] = l2_mean(diff)
    out["chunk_diff_abs_mean"] = abs_mean(diff)

    out["chunk_accel_l2_mean"] = l2_mean(accel)
    out["chunk_accel_abs_mean"] = abs_mean(accel)

    out["chunk_jerk_l2_mean"] = l2_mean(jerk)
    out["chunk_jerk_abs_mean"] = abs_mean(jerk)

    flip_per_dim, flip_mean = sign_flip_rate(actions)
    out["chunk_sign_flip_mean"] = flip_mean

    out["chunk_diff_abs_per_dim"] = per_dim_abs_mean(diff)
    out["chunk_accel_abs_per_dim"] = per_dim_abs_mean(accel)
    out["chunk_jerk_abs_per_dim"] = per_dim_abs_mean(jerk)
    out["chunk_jerk_max_per_dim"] = per_dim_abs_max(jerk)
    out["chunk_sign_flip_per_dim"] = flip_per_dim

    return out


def overlap_metrics(prev_chunk, curr_chunk):
    """
    prev_chunk: (T, A), predicted at frame t
    curr_chunk: (T, A), predicted at frame t+1

    overlap_:
      prev_chunk[1:] corresponds to future t+1 ... t+99
      curr_chunk[:-1] corresponds to future t+1 ... t+99

      overlap_delta = curr_chunk[:-1] - prev_chunk[1:]
    """
    overlap_delta = curr_chunk[:-1] - prev_chunk[1:]  # (T-1, A)

    out = {}
    out["overlap_l2_mean"] = l2_mean(overlap_delta)
    out["overlap_abs_mean"] = abs_mean(overlap_delta)

    # overlap delta가 horizon 안에서 얼마나 울퉁불퉁한지도 확인
    if overlap_delta.shape[0] >= 2:
        odiff = overlap_delta[1:] - overlap_delta[:-1]
        out["overlap_delta_diff_l2_mean"] = l2_mean(odiff)
        out["overlap_delta_diff_abs_mean"] = abs_mean(odiff)
    else:
        out["overlap_delta_diff_l2_mean"] = 0.0
        out["overlap_delta_diff_abs_mean"] = 0.0

    if overlap_delta.shape[0] >= 3:
        oaccel = overlap_delta[2:] - 2.0 * overlap_delta[1:-1] + overlap_delta[:-2]
        out["overlap_delta_accel_l2_mean"] = l2_mean(oaccel)
        out["overlap_delta_accel_abs_mean"] = abs_mean(oaccel)
    else:
        out["overlap_delta_accel_l2_mean"] = 0.0
        out["overlap_delta_accel_abs_mean"] = 0.0

    out["overlap_abs_per_dim"] = per_dim_abs_mean(overlap_delta)
    out["overlap_max_per_dim"] = per_dim_abs_max(overlap_delta)

    return out


def mean_scalar(metrics_list, key):
    vals = [m[key] for m in metrics_list if key in m]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def mean_array(metrics_list, key):
    vals = [m[key] for m in metrics_list if key in m]
    if len(vals) == 0:
        return None
    return np.mean(vals, axis=0)


def print_scalar_block(title, metrics_list, keys):
    print("\n" + "=" * 100)
    print(title)
    for k in keys:
        v = mean_scalar(metrics_list, k)
        if v is None:
            print(f"{k:32s} = N/A")
        else:
            print(f"{k:32s} = {v:.8f}")


def print_per_dim_block(title, metrics_list):
    print("\n" + "=" * 100)
    print(title)

    diff = mean_array(metrics_list, "chunk_diff_abs_per_dim")
    accel = mean_array(metrics_list, "chunk_accel_abs_per_dim")
    jerk = mean_array(metrics_list, "chunk_jerk_abs_per_dim")
    jerk_max = mean_array(metrics_list, "chunk_jerk_max_per_dim")
    flip = mean_array(metrics_list, "chunk_sign_flip_per_dim")

    if diff is None:
        print("No per-dim chunk metrics.")
        return

    print("dim | diff_abs | accel_abs | jerk_abs | jerk_max | sign_flip")
    print("-" * 90)

    for d in range(len(diff)):
        print(
            f"{d:3d} | "
            f"{diff[d]:.8f} | "
            f"{accel[d]:.8f} | "
            f"{jerk[d]:.8f} | "
            f"{jerk_max[d]:.8f} | "
            f"{flip[d]:.4f}"
        )

    order = np.argsort(-jerk)
    print("\n[Top jitter dims by chunk jerk_abs]")
    for rank, d in enumerate(order[:5], 1):
        print(
            f"rank {rank}: dim={d}, "
            f"jerk_abs={jerk[d]:.8f}, "
            f"jerk_max={jerk_max[d]:.8f}, "
            f"accel_abs={accel[d]:.8f}, "
            f"diff_abs={diff[d]:.8f}, "
            f"sign_flip={flip[d]:.4f}"
        )


def print_overlap_per_dim(metrics_list):
    print("\n" + "=" * 100)
    print("[Overlap per-dim metrics]")

    overlap = mean_array(metrics_list, "overlap_abs_per_dim")
    overlap_max = mean_array(metrics_list, "overlap_max_per_dim")

    if overlap is None:
        print("No overlap metrics. Maybe no consecutive frames found.")
        return

    print("dim | overlap_abs | overlap_max")
    print("-" * 55)

    for d in range(len(overlap)):
        print(f"{d:3d} | {overlap[d]:.8f} | {overlap_max[d]:.8f}")

    order = np.argsort(-overlap)
    print("\n[Top unstable dims by overlap_abs]")
    for rank, d in enumerate(order[:5], 1):
        print(
            f"rank {rank}: dim={d}, "
            f"overlap_abs={overlap[d]:.8f}, "
            f"overlap_max={overlap_max[d]:.8f}"
        )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--num_frames", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--compare_raw_refined",
        action="store_true",
        help="If action_conv_refiner exists, also compare action_head raw chunk vs final refined chunk.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy_path = Path(args.policy_path)

    print("=" * 100)
    print("[Analyze Chunk / Overlap Smoothness]")
    print(f"policy_path          = {policy_path}")
    print(f"dataset_repo_id      = {args.dataset_repo_id}")
    print(f"num_frames           = {args.num_frames}")
    print(f"device               = {device}")
    print(f"compare_raw_refined  = {args.compare_raw_refined}")
    print("=" * 100)

    policy = ACMPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    print("[CONFIG]")
    print(f"use_mamba                  = {getattr(policy.config, 'use_mamba', None)}")
    print(f"use_action_conv_refiner    = {getattr(policy.config, 'use_action_conv_refiner', None)}")
    print(f"action_refiner_kernel_size = {getattr(policy.config, 'action_refiner_kernel_size', None)}")
    print(f"delta_magnitude_weight     = {getattr(policy.config, 'delta_magnitude_weight', None)}")
    print(f"delta_smoothness_weight    = {getattr(policy.config, 'delta_smoothness_weight', None)}")
    print(f"action_jerk_weight         = {getattr(policy.config, 'action_jerk_weight', None)}")
    print(f"chunk_size                 = {getattr(policy.config, 'chunk_size', None)}")
    print(f"n_action_steps             = {getattr(policy.config, 'n_action_steps', None)}")
    print("=" * 100)

    dataset = LeRobotDataset(args.dataset_repo_id)

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        dataset_stats=dataset.meta.stats,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    chunk_list = []
    overlap_list = []

    raw_chunk_list = []
    raw_refined_delta_list = []

    prev_ep = None
    prev_frame = None
    prev_chunk = None

    n_overlap = 0
    n_seen = 0

    for batch in tqdm(loader, total=min(args.num_frames, len(dataset))):
        if n_seen >= args.num_frames:
            break

        n_seen += 1

        # episode / frame 정보 가져오기
        # LeRobotDataset 버전에 따라 key 이름이 다를 수 있어 후보를 여러 개 둠.
        ep_key = None
        for k in ["episode_index", "episode_idx", "episode"]:
            if k in batch:
                ep_key = k
                break

        frame_key = None
        for k in ["frame_index", "frame_idx", "index"]:
            if k in batch:
                frame_key = k
                break

        curr_ep = get_scalar(batch[ep_key]) if ep_key is not None else 0
        curr_frame = get_scalar(batch[frame_key]) if frame_key is not None else n_seen - 1

        batch = to_device(batch, device)
        batch = preprocessor(batch)

        if policy.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

        captured = {}

        handle = None
        if args.compare_raw_refined and hasattr(policy.model, "action_head"):
            def hook_action_head(module, inputs, output):
                captured["raw_actions"] = output.detach()

            handle = policy.model.action_head.register_forward_hook(hook_action_head)

        actions = policy.model(batch)[0]  # (1, T, A)

        if handle is not None:
            handle.remove()

        final_chunk = actions[0].detach()  # (T, A)
        chunk_list.append(chunk_metrics(final_chunk))

        if "raw_actions" in captured:
            raw_chunk = captured["raw_actions"][0].detach()
            raw_chunk_list.append(chunk_metrics(raw_chunk))

            delta = final_chunk - raw_chunk
            raw_refined_delta_list.append(
                {
                    "raw_refined_delta_l2_mean": l2_mean(delta),
                    "raw_refined_delta_abs_mean": abs_mean(delta),
                }
            )

        # overlap_: 같은 episode에서 frame이 연속일 때만 계산
        is_consecutive = (
            prev_chunk is not None
            and curr_ep == prev_ep
            and curr_frame == prev_frame + 1
        )

        if is_consecutive:
            overlap_list.append(overlap_metrics(prev_chunk, final_chunk))
            n_overlap += 1

        prev_ep = curr_ep
        prev_frame = curr_frame
        prev_chunk = final_chunk

    print("\n" + "=" * 100)
    print(f"[Frames used] n_seen={n_seen}, n_overlap={n_overlap}")
    print("=" * 100)

    chunk_keys = [
        "chunk_diff_l2_mean",
        "chunk_diff_abs_mean",
        "chunk_accel_l2_mean",
        "chunk_accel_abs_mean",
        "chunk_jerk_l2_mean",
        "chunk_jerk_abs_mean",
        "chunk_sign_flip_mean",
    ]

    overlap_keys = [
        "overlap_l2_mean",
        "overlap_abs_mean",
        "overlap_delta_diff_l2_mean",
        "overlap_delta_diff_abs_mean",
        "overlap_delta_accel_l2_mean",
        "overlap_delta_accel_abs_mean",
    ]

    print_scalar_block("[FINAL chunk_ metrics]", chunk_list, chunk_keys)
    print_per_dim_block("[FINAL chunk_ per-dim metrics]", chunk_list)

    print_scalar_block("[FINAL overlap_ metrics]", overlap_list, overlap_keys)
    print_overlap_per_dim(overlap_list)

    if len(raw_chunk_list) > 0:
        print_scalar_block("[RAW chunk_ metrics before action_conv_refiner]", raw_chunk_list, chunk_keys)

        delta_keys = [
            "raw_refined_delta_l2_mean",
            "raw_refined_delta_abs_mean",
        ]
        print_scalar_block("[RAW -> FINAL delta metrics]", raw_refined_delta_list, delta_keys)

        print("\n" + "=" * 100)
        print("[FINAL - RAW chunk_ change]")
        for k in chunk_keys:
            final_v = mean_scalar(chunk_list, k)
            raw_v = mean_scalar(raw_chunk_list, k)
            diff = final_v - raw_v
            rel = diff / (raw_v + 1e-12) * 100
            print(f"{k:32s} = {diff:+.8f} ({rel:+.4f}%)")

    print("\n" + "=" * 100)
    print("[Interpretation guide]")
    print("chunk_  : one predicted chunk 내부의 diff / accel / jerk")
    print("overlap_: consecutive chunks가 같은 미래 timestep을 얼마나 다르게 예측하는지")
    print("high chunk_jerk + high sign_flip -> chunk 내부 부들거림 가능성")
    print("high overlap_ -> observation이 한 step 바뀔 때 예측 계획이 불안정한 것")
    print("=" * 100)


if __name__ == "__main__":
    main()
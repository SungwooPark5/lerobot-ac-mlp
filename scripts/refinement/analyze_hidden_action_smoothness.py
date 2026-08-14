#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import OBS_IMAGES


def load_policy(policy_path: Path):
    cfg_path = policy_path / "config.json"
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    policy_type = cfg.get("type", cfg.get("policy_type", None))

    if policy_type == "acm":
        policy = ACMPolicy.from_pretrained(policy_path)
    elif policy_type == "act":
        policy = ACTPolicy.from_pretrained(policy_path)
    else:
        raise ValueError(f"Unknown policy type from config.json: {policy_type}")

    return policy, policy_type


def to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def get_scalar(x):
    if torch.is_tensor(x):
        return int(x.flatten()[0].item())
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


def l2_mean(x):
    return torch.linalg.norm(x, dim=-1).mean().item()


def l2_sqrt_dim_mean(x):
    # hidden dim이 512처럼 클 때 ACT/ACM 비교용으로 dim scale을 보정
    return (torch.linalg.norm(x, dim=-1) / np.sqrt(x.shape[-1])).mean().item()


def abs_mean(x):
    return x.abs().mean().item()


def sign_flip_rate(seq, eps=1e-6):
    """
    seq: (T, D) or (B, T, D)
    """
    if seq.ndim == 2:
        seq = seq.unsqueeze(0)

    diff = seq[:, 1:] - seq[:, :-1]
    if diff.shape[1] < 2:
        return 0.0

    d0 = diff[:, :-1]
    d1 = diff[:, 1:]

    valid = (d0.abs() > eps) & (d1.abs() > eps)
    flips = (d0 * d1 < 0) & valid

    denom = valid.sum().clamp(min=1)
    return (flips.sum().float() / denom.float()).item()


def seq_metrics(seq, prefix):
    """
    seq: (B, T, D) or (T, D)
    """
    if seq.ndim == 2:
        seq = seq.unsqueeze(0)

    out = {}

    diff = seq[:, 1:] - seq[:, :-1]
    accel = seq[:, 2:] - 2.0 * seq[:, 1:-1] + seq[:, :-2]
    jerk = seq[:, 3:] - 3.0 * seq[:, 2:-1] + 3.0 * seq[:, 1:-2] - seq[:, :-3]

    out[f"{prefix}_diff_l2"] = l2_mean(diff)
    out[f"{prefix}_diff_l2_sqrtdim"] = l2_sqrt_dim_mean(diff)
    out[f"{prefix}_diff_abs"] = abs_mean(diff)

    out[f"{prefix}_accel_l2"] = l2_mean(accel)
    out[f"{prefix}_accel_l2_sqrtdim"] = l2_sqrt_dim_mean(accel)
    out[f"{prefix}_accel_abs"] = abs_mean(accel)

    out[f"{prefix}_jerk_l2"] = l2_mean(jerk)
    out[f"{prefix}_jerk_l2_sqrtdim"] = l2_sqrt_dim_mean(jerk)
    out[f"{prefix}_jerk_abs"] = abs_mean(jerk)

    out[f"{prefix}_sign_flip"] = sign_flip_rate(seq)

    return out


def overlap_metrics(prev_seq, curr_seq, prefix):
    """
    prev_seq: (T, D), predicted at frame t
    curr_seq: (T, D), predicted at frame t+1

    compare:
      prev_seq[1:] vs curr_seq[:-1]
    """
    delta = curr_seq[:-1] - prev_seq[1:]

    out = {}
    out[f"{prefix}_overlap_l2"] = l2_mean(delta)
    out[f"{prefix}_overlap_l2_sqrtdim"] = l2_sqrt_dim_mean(delta)
    out[f"{prefix}_overlap_abs"] = abs_mean(delta)

    if delta.shape[0] >= 2:
        dd = delta[1:] - delta[:-1]
        out[f"{prefix}_overlap_delta_diff_l2"] = l2_mean(dd)
        out[f"{prefix}_overlap_delta_diff_l2_sqrtdim"] = l2_sqrt_dim_mean(dd)
        out[f"{prefix}_overlap_delta_diff_abs"] = abs_mean(dd)
    else:
        out[f"{prefix}_overlap_delta_diff_l2"] = 0.0
        out[f"{prefix}_overlap_delta_diff_l2_sqrtdim"] = 0.0
        out[f"{prefix}_overlap_delta_diff_abs"] = 0.0

    return out


def mean_metric(metrics, key):
    vals = [m[key] for m in metrics if key in m]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def print_block(title, metrics, keys):
    print("\n" + "=" * 100)
    print(title)
    for k in keys:
        v = mean_metric(metrics, k)
        if v is None:
            print(f"{k:42s} = N/A")
        else:
            print(f"{k:42s} = {v:.8f}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--num_frames", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    policy_path = Path(args.policy_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("[Analyze Hidden vs Action Smoothness]")
    print(f"policy_path     = {policy_path}")
    print(f"dataset_repo_id = {args.dataset_repo_id}")
    print(f"num_frames      = {args.num_frames}")
    print(f"device          = {device}")
    print("=" * 100)

    policy, policy_type = load_policy(policy_path)
    policy.to(device)
    policy.eval()

    print("[CONFIG]")
    print(f"policy_type              = {policy_type}")
    print(f"use_mamba                = {getattr(policy.config, 'use_mamba', None)}")
    print(f"use_action_conv_refiner  = {getattr(policy.config, 'use_action_conv_refiner', None)}")
    print(f"action_jerk_weight       = {getattr(policy.config, 'action_jerk_weight', None)}")
    print(f"chunk_size               = {getattr(policy.config, 'chunk_size', None)}")
    print(f"n_action_steps           = {getattr(policy.config, 'n_action_steps', None)}")
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

    chunk_metrics = []
    overlap_metrics_list = []

    prev_ep = None
    prev_frame = None
    prev_hidden = None
    prev_raw_action = None
    prev_final_action = None

    n_seen = 0
    n_overlap = 0

    for batch in tqdm(loader, total=min(args.num_frames, len(dataset))):
        if n_seen >= args.num_frames:
            break

        ep_key = next((k for k in ["episode_index", "episode_idx", "episode"] if k in batch), None)
        frame_key = next((k for k in ["frame_index", "frame_idx", "index"] if k in batch), None)

        curr_ep = get_scalar(batch[ep_key]) if ep_key is not None else 0
        curr_frame = get_scalar(batch[frame_key]) if frame_key is not None else n_seen

        batch = to_device(batch, device)
        batch = preprocessor(batch)

        if policy.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

        captured = {}

        def action_head_hook(module, inputs, output):
            # inputs[0]: decoder_out before action_head, usually (B, T, hidden_dim)
            # output: raw action before optional refiner, usually (B, T, action_dim)
            captured["hidden"] = inputs[0].detach()
            captured["raw_action"] = output.detach()

        handle = policy.model.action_head.register_forward_hook(action_head_hook)
        final_action = policy.model(batch)[0].detach()
        handle.remove()

        if "hidden" not in captured:
            raise RuntimeError("Failed to capture action_head input. Check policy.model.action_head.")

        hidden = captured["hidden"][0]          # (T, hidden_dim)
        raw_action = captured["raw_action"][0]  # (T, action_dim)
        final_action = final_action[0]          # (T, action_dim)

        m = {}
        m.update(seq_metrics(hidden, "hidden"))
        m.update(seq_metrics(raw_action, "raw_action"))
        m.update(seq_metrics(final_action, "final_action"))
        chunk_metrics.append(m)

        is_consecutive = (
            prev_hidden is not None
            and curr_ep == prev_ep
            and curr_frame == prev_frame + 1
        )

        if is_consecutive:
            om = {}
            om.update(overlap_metrics(prev_hidden, hidden, "hidden"))
            om.update(overlap_metrics(prev_raw_action, raw_action, "raw_action"))
            om.update(overlap_metrics(prev_final_action, final_action, "final_action"))
            overlap_metrics_list.append(om)
            n_overlap += 1

        prev_ep = curr_ep
        prev_frame = curr_frame
        prev_hidden = hidden
        prev_raw_action = raw_action
        prev_final_action = final_action

        n_seen += 1

    print("\n" + "=" * 100)
    print(f"[Frames used] n_seen={n_seen}, n_overlap={n_overlap}")
    print("=" * 100)

    chunk_keys = [
        "hidden_diff_l2_sqrtdim",
        "hidden_accel_l2_sqrtdim",
        "hidden_jerk_l2_sqrtdim",
        "hidden_sign_flip",
        "raw_action_diff_l2",
        "raw_action_accel_l2",
        "raw_action_jerk_l2",
        "raw_action_sign_flip",
        "final_action_diff_l2",
        "final_action_accel_l2",
        "final_action_jerk_l2",
        "final_action_sign_flip",
    ]

    overlap_keys = [
        "hidden_overlap_l2_sqrtdim",
        "hidden_overlap_delta_diff_l2_sqrtdim",
        "raw_action_overlap_l2",
        "raw_action_overlap_delta_diff_l2",
        "final_action_overlap_l2",
        "final_action_overlap_delta_diff_l2",
    ]

    print_block("[Chunk smoothness: hidden vs action]", chunk_metrics, chunk_keys)
    print_block("[Overlap smoothness: hidden vs action]", overlap_metrics_list, overlap_keys)

    print("\n" + "=" * 100)
    print("[Interpretation]")
    print("hidden_*_sqrtdim: hidden_dim=512 scale 보정한 값. ACT/ACM hidden 비교용.")
    print("raw_action_*: action_head output. refiner OFF면 final_action과 같음.")
    print("final_action_*: 실제 policy output. refiner ON이면 refined action.")
    print("")
    print("판정:")
    print("1) ACM hidden부터 ACT보다 크면 decoder representation 단계 문제.")
    print("2) hidden은 비슷한데 raw/final action만 크면 action_head/action regression 문제.")
    print("3) hidden overlap부터 크면 observation 변화에 따른 plan consistency 문제.")
    print("=" * 100)


if __name__ == '__main__':
    main()
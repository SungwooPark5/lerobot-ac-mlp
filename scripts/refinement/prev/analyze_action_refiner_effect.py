#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES


def l2_mean(x):
    # x: (..., D)
    return torch.linalg.norm(x, dim=-1).mean().item()


def abs_mean(x):
    return x.abs().mean().item()


def smooth_stats(actions):
    """
    actions: (B, T, A)
    """
    out = {}

    diff = actions[:, 1:] - actions[:, :-1]
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    jerk = (
        actions[:, 3:]
        - 3 * actions[:, 2:-1]
        + 3 * actions[:, 1:-2]
        - actions[:, :-3]
    )

    out["diff_l2_mean"] = l2_mean(diff)
    out["diff_abs_mean"] = abs_mean(diff)

    out["accel_l2_mean"] = l2_mean(accel)
    out["accel_abs_mean"] = abs_mean(accel)

    out["jerk_l2_mean"] = l2_mean(jerk)
    out["jerk_abs_mean"] = abs_mean(jerk)

    return out


@torch.no_grad()
def get_raw_and_refined(policy, batch):
    """
    Re-run the ACM forward path manually enough to capture:
      raw_actions = action_head(decoder_out)
      refined_actions = action_conv_refiner(raw_actions)

    This assumes ACMPolicy with action_conv_refiner.
    """
    model = policy.model
    model.eval()

    # batch image handling: same as policy.predict_action_chunk
    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

    # We need to mostly copy ACM.forward(), but stop around action_head.
    # Easier: temporarily disable refiner to get raw? No.
    # Better: attach a forward hook on action_head output.
    captured = {}

    def hook_action_head(module, inputs, output):
        captured["raw_actions"] = output.detach()

    handle = model.action_head.register_forward_hook(hook_action_head)

    refined_actions, _ = model(batch)

    handle.remove()

    raw_actions = captured["raw_actions"]
    refined_actions = refined_actions.detach()

    delta = refined_actions - raw_actions

    return raw_actions, refined_actions, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="Path to pretrained_model checkpoint directory.",
    )
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="lerobot/aloha_sim_transfer_cube_human",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("[Analyze Action Conv Refiner Effect]")
    print(f"policy_path     = {args.policy_path}")
    print(f"dataset_repo_id = {args.dataset_repo_id}")
    print(f"batch_size      = {args.batch_size}")
    print(f"num_batches     = {args.num_batches}")
    print(f"device          = {device}")
    print("=" * 100)

    policy_path = Path(args.policy_path)

    policy = ACMPolicy.from_pretrained(policy_path)
    policy.to(device)
    policy.eval()

    print("[CONFIG]")
    print(f"use_mamba                  = {getattr(policy.config, 'use_mamba', None)}")
    print(f"use_action_conv_refiner    = {getattr(policy.config, 'use_action_conv_refiner', None)}")
    print(f"action_refiner_kernel_size = {getattr(policy.config, 'action_refiner_kernel_size', None)}")
    print(f"action_refiner_alpha_init  = {getattr(policy.config, 'action_refiner_alpha_init', None)}")
    print(f"delta_magnitude_weight     = {getattr(policy.config, 'delta_magnitude_weight', None)}")
    print(f"delta_smoothness_weight    = {getattr(policy.config, 'delta_smoothness_weight', None)}")
    print(f"action_jerk_weight         = {getattr(policy.config, 'action_jerk_weight', None)}")

    if not getattr(policy.config, "use_action_conv_refiner", False):
        print("[WARN] use_action_conv_refiner=False. This checkpoint does not use action conv refiner.")

    if getattr(policy.model, "action_conv_refiner", None) is None:
        print("[WARN] policy.model.action_conv_refiner is None.")

    dataset = LeRobotDataset(args.dataset_repo_id)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_path,
        dataset_stats=dataset.meta.stats,
    )

    raw_stats_list = []
    refined_stats_list = []
    delta_abs = []
    delta_l2 = []
    raw_refined_l1 = []
    raw_refined_l2 = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    for i, batch in enumerate(tqdm(loader, total=args.num_batches)):
        if i >= args.num_batches:
            break

        batch = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        batch = preprocessor(batch)

        raw_actions, refined_actions, delta = get_raw_and_refined(policy, batch)

        raw_stats_list.append(smooth_stats(raw_actions))
        refined_stats_list.append(smooth_stats(refined_actions))

        delta_abs.append(delta.abs().mean().item())
        delta_l2.append(torch.linalg.norm(delta, dim=-1).mean().item())

        raw_refined_l1.append(F.l1_loss(refined_actions, raw_actions).item())
        raw_refined_l2.append(torch.linalg.norm(refined_actions - raw_actions, dim=-1).mean().item())

    def mean_dict(dicts):
        keys = dicts[0].keys()
        return {k: float(np.mean([d[k] for d in dicts])) for k in keys}

    raw_s = mean_dict(raw_stats_list)
    refined_s = mean_dict(refined_stats_list)

    print("\n" + "=" * 100)
    print("[Delta: refined_actions - raw_actions]")
    print(f"delta_abs_mean        = {np.mean(delta_abs):.8f}")
    print(f"delta_l2_mean         = {np.mean(delta_l2):.8f}")
    print(f"raw_refined_l1        = {np.mean(raw_refined_l1):.8f}")
    print(f"raw_refined_l2        = {np.mean(raw_refined_l2):.8f}")

    print("\n" + "=" * 100)
    print("[Raw action smoothness]")
    for k, v in raw_s.items():
        print(f"{k:20s} = {v:.8f}")

    print("\n" + "=" * 100)
    print("[Refined action smoothness]")
    for k, v in refined_s.items():
        print(f"{k:20s} = {v:.8f}")

    print("\n" + "=" * 100)
    print("[Refined - Raw smoothness change]")
    for k in raw_s:
        diff = refined_s[k] - raw_s[k]
        rel = diff / (raw_s[k] + 1e-12) * 100
        print(f"{k:20s} = {diff:+.8f} ({rel:+.4f}%)")

    print("\n" + "=" * 100)
    print("[Interpretation]")
    if np.mean(delta_l2) < 1e-4:
        print("Refiner is almost not changing actions. delta_l2_mean < 1e-4.")
    elif np.mean(delta_l2) < 1e-3:
        print("Refiner changes actions very weakly. delta_l2_mean < 1e-3.")
    else:
        print("Refiner is changing actions noticeably.")

    if refined_s["jerk_l2_mean"] < raw_s["jerk_l2_mean"]:
        print("Refined actions have lower jerk than raw actions.")
    else:
        print("Refined actions do NOT reduce jerk compared to raw actions.")

    print("=" * 100)


if __name__ == "__main__":
    main()
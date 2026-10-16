#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.utils.constants import OBS_IMAGES


def smoothness_metrics(actions):
    actions = actions.float()
    diff = actions[:, 1:] - actions[:, :-1]
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    jerk = actions[:, 3:] - 3 * actions[:, 2:-1] + 3 * actions[:, 1:-2] - actions[:, :-3]

    def stat(x):
        l2 = torch.linalg.norm(x, dim=-1)
        return {
            "l2_mean": l2.mean().item(),
            "l2_median": l2.median().item(),
            "l2_max": l2.max().item(),
            "abs_mean": x.abs().mean().item(),
        }

    return {"diff": stat(diff), "accel": stat(accel), "jerk": stat(jerk)}


def print_metrics(name, metrics):
    print(f"\n[{name}]")
    for k, v in metrics.items():
        print(
            f"{k:>6s} | "
            f"l2_mean={v['l2_mean']:.6f} | "
            f"l2_median={v['l2_median']:.6f} | "
            f"l2_max={v['l2_max']:.6f} | "
            f"abs_mean={v['abs_mean']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"[LOAD POLICY] {args.policy_path}")
    policy = ACMPolicy.from_pretrained(Path(args.policy_path))
    policy.to(args.device)
    policy.eval()

    print(f"[HAS ACTION CONV REFINER] {policy.model.action_conv_refiner is not None}")

    dataset = LeRobotDataset(args.dataset_repo_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    raw_chunks = []
    refined_chunks = []

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader)):
            if args.max_batches > 0 and bi >= args.max_batches:
                break

            batch = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            if policy.config.image_features:
                batch = dict(batch)
                batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

            refiner = policy.model.action_conv_refiner

            # before refiner
            policy.model.action_conv_refiner = None
            raw = policy.model(batch)[0]

            # after refiner
            policy.model.action_conv_refiner = refiner
            refined = policy.model(batch)[0]

            raw_chunks.append(raw.detach().cpu())
            refined_chunks.append(refined.detach().cpu())

    raw_chunks = torch.cat(raw_chunks, dim=0)
    refined_chunks = torch.cat(refined_chunks, dim=0)
    delta = refined_chunks - raw_chunks
    delta_diff = delta[:, 1:] - delta[:, :-1]

    print(f"[RAW CHUNKS]     {raw_chunks.shape}")
    print(f"[REFINED CHUNKS] {refined_chunks.shape}")

    raw_metrics = smoothness_metrics(raw_chunks)
    refined_metrics = smoothness_metrics(refined_chunks)
    delta_metrics = smoothness_metrics(delta)

    print_metrics("Before action conv refiner", raw_metrics)
    print_metrics("After action conv refiner", refined_metrics)
    print_metrics("Delta smoothness", delta_metrics)

    print("\n[After / Before ratio]")
    for k in ["diff", "accel", "jerk"]:
        r = refined_metrics[k]["l2_mean"] / raw_metrics[k]["l2_mean"]
        print(f"{k:>6s} l2_mean ratio = {r:.4f}")

    print("\n[Delta magnitude]")
    print(f"delta abs mean      = {delta.abs().mean().item():.6f}")
    print(f"delta l2 mean       = {torch.linalg.norm(delta, dim=-1).mean().item():.6f}")
    print(f"delta diff abs mean = {delta_diff.abs().mean().item():.6f}")
    print(f"delta diff l2 mean  = {torch.linalg.norm(delta_diff, dim=-1).mean().item():.6f}")

    if refiner is not None and hasattr(refiner, "alpha"):
        print("\n[Alpha]")
        print(f"alpha raw  = {refiner.alpha.item():.6f}")
        print(f"tanh alpha = {torch.tanh(refiner.alpha).item():.6f}")


if __name__ == "__main__":
    main()

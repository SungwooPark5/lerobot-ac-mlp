#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.acm.modeling_acm import ACMPolicy


def smoothness_metrics(actions: torch.Tensor):
    """
    actions: (N, T, D)
    """
    actions = actions.float()

    diff = actions[:, 1:] - actions[:, :-1]
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    jerk = (
        actions[:, 3:]
        - 3 * actions[:, 2:-1]
        + 3 * actions[:, 1:-2]
        - actions[:, :-3]
    )

    def stat(x):
        l2 = torch.linalg.norm(x, dim=-1)
        abs_mean = x.abs().mean()
        return {
            "l2_mean": l2.mean().item(),
            "l2_median": l2.median().item(),
            "l2_max": l2.max().item(),
            "abs_mean": abs_mean.item(),
        }

    return {
        "diff": stat(diff),
        "accel": stat(accel),
        "jerk": stat(jerk),
    }


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

    policy_path = Path(args.policy_path)

    print(f"[LOAD POLICY] {policy_path}")
    policy = ACMPolicy.from_pretrained(policy_path)
    policy.to(args.device)
    policy.eval()

    print(f"[LOAD DATASET] {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    all_chunks = []

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader)):
            if args.max_batches > 0 and bi >= args.max_batches:
                break

            batch = {
                k: v.to(args.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            pred_chunk = policy.predict_action_chunk(batch)  # (B, T, D)
            all_chunks.append(pred_chunk.detach().cpu())

    chunks = torch.cat(all_chunks, dim=0)
    print(f"[PRED CHUNKS] {chunks.shape}")

    metrics = smoothness_metrics(chunks)
    print_metrics("Predicted chunk smoothness", metrics)


if __name__ == "__main__":
    main()
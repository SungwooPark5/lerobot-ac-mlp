#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES


def to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def l2_mean(x):
    return torch.linalg.norm(x, dim=-1).mean().item()


def abs_mean(x):
    return x.abs().mean().item()


def sign_flip_rate(actions, eps=1e-6):
    diff = actions[:, 1:] - actions[:, :-1]
    d0 = diff[:, :-1]
    d1 = diff[:, 1:]

    valid = (d0.abs() > eps) & (d1.abs() > eps)
    flips = (d0 * d1 < 0) & valid

    denom = valid.sum().clamp(min=1)
    return (flips.sum().float() / denom.float()).item()


def chunk_metrics(actions):
    """
    actions: (B, T, A)
    """
    diff = actions[:, 1:] - actions[:, :-1]
    accel = actions[:, 2:] - 2.0 * actions[:, 1:-1] + actions[:, :-2]
    jerk = (
        actions[:, 3:]
        - 3.0 * actions[:, 2:-1]
        + 3.0 * actions[:, 1:-2]
        - actions[:, :-3]
    )

    return {
        "diff_l2": l2_mean(diff),
        "diff_abs": abs_mean(diff),
        "accel_l2": l2_mean(accel),
        "accel_abs": abs_mean(accel),
        "jerk_l2": l2_mean(jerk),
        "jerk_abs": abs_mean(jerk),
        "sign_flip": sign_flip_rate(actions),
    }


def overlap_metrics(prev_chunk, curr_chunk):
    """
    prev_chunk, curr_chunk: (T, A)
    """
    delta = curr_chunk[:-1] - prev_chunk[1:]
    return {
        "overlap_l2": torch.linalg.norm(delta, dim=-1).mean().item(),
        "overlap_abs": delta.abs().mean().item(),
    }


def mean_dict(dicts):
    out = {}
    keys = dicts[0].keys()
    for k in keys:
        out[k] = float(np.mean([d[k] for d in dicts]))
    return out


def get_scalar(x):
    if torch.is_tensor(x):
        return int(x.flatten()[0].item())
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act_path", type=str, required=True)
    parser.add_argument("--acm_path", type=str, required=True)
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--num_frames", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    alphas = [float(x) for x in args.alphas.split(",")]

    act_path = Path(args.act_path)
    acm_path = Path(args.acm_path)

    print("=" * 100)
    print("[ACT + ACM fixed ensemble offline analysis]")
    print(f"ACT path   = {act_path}")
    print(f"ACM path   = {acm_path}")
    print(f"dataset    = {args.dataset_repo_id}")
    print(f"num_frames = {args.num_frames}")
    print(f"alphas     = {alphas}")
    print(f"device     = {device}")
    print("=" * 100)

    act = ACTPolicy.from_pretrained(act_path).to(device).eval()
    acm = ACMPolicy.from_pretrained(acm_path).to(device).eval()

    dataset = LeRobotDataset(args.dataset_repo_id)

    act_pre, _ = make_pre_post_processors(
        act.config,
        pretrained_path=act_path,
        dataset_stats=dataset.meta.stats,
    )
    acm_pre, _ = make_pre_post_processors(
        acm.config,
        pretrained_path=acm_path,
        dataset_stats=dataset.meta.stats,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    metrics_by_alpha = {a: [] for a in alphas}
    l1_by_alpha = {a: [] for a in alphas}

    prev_by_alpha = {a: None for a in alphas}
    prev_ep = None
    prev_frame = None
    overlap_by_alpha = {a: [] for a in alphas}

    n_seen = 0
    n_overlap = 0

    for batch in tqdm(loader):
        if n_seen >= args.num_frames:
            break

        bsz = next(v.shape[0] for v in batch.values() if torch.is_tensor(v))
        remain = args.num_frames - n_seen
        if bsz > remain:
            batch = {
                k: v[:remain] if torch.is_tensor(v) and v.shape[0] == bsz else v
                for k, v in batch.items()
            }
            bsz = remain

        batch_cpu = batch
        batch = to_device(batch, device)

        act_batch = act_pre(dict(batch))
        acm_batch = acm_pre(dict(batch))

        if act.config.image_features:
            act_batch = dict(act_batch)
            act_batch[OBS_IMAGES] = [act_batch[key] for key in act.config.image_features]

        if acm.config.image_features:
            acm_batch = dict(acm_batch)
            acm_batch[OBS_IMAGES] = [acm_batch[key] for key in acm.config.image_features]

        act_chunk = act.model(act_batch)[0]
        acm_chunk = acm.model(acm_batch)[0]

        target = acm_batch[ACTION]

        for alpha in alphas:
            mix = alpha * act_chunk + (1.0 - alpha) * acm_chunk

            metrics_by_alpha[alpha].append(chunk_metrics(mix))

            # LeRobotDataset may provide ACTION as either:
            #   (B, A)    : current expert action only
            #   (B, T, A) : expert action chunk
            # For offline alpha sweep, if only current action exists,
            # compare it against the first action of the predicted chunk.
            if target.ndim == 2:
                l1 = torch.nn.functional.l1_loss(
                    mix[:, 0, :], target, reduction="none"
                ).mean().item()
            elif target.ndim == 3:
                l1 = torch.nn.functional.l1_loss(
                    mix, target, reduction="none"
                ).mean().item()
            else:
                raise RuntimeError(f"Unexpected ACTION target shape: {target.shape}")

            l1_by_alpha[alpha].append(l1)

        # overlap은 batch_size>1일 때 batch 내부 연속성 처리 귀찮아서 첫 sample 기준만 사용
        ep_key = next((k for k in ["episode_index", "episode_idx", "episode"] if k in batch_cpu), None)
        frame_key = next((k for k in ["frame_index", "frame_idx", "index"] if k in batch_cpu), None)

        for i in range(bsz):
            curr_ep = get_scalar(batch_cpu[ep_key][i]) if ep_key is not None and torch.is_tensor(batch_cpu[ep_key]) else 0
            curr_frame = get_scalar(batch_cpu[frame_key][i]) if frame_key is not None and torch.is_tensor(batch_cpu[frame_key]) else n_seen + i

            is_consecutive = prev_ep is not None and curr_ep == prev_ep and curr_frame == prev_frame + 1

            for alpha in alphas:
                mix_i = (alpha * act_chunk[i] + (1.0 - alpha) * acm_chunk[i]).detach()

                if is_consecutive and prev_by_alpha[alpha] is not None:
                    overlap_by_alpha[alpha].append(overlap_metrics(prev_by_alpha[alpha], mix_i))
                    if alpha == alphas[0]:
                        n_overlap += 1

                prev_by_alpha[alpha] = mix_i

            prev_ep = curr_ep
            prev_frame = curr_frame

        n_seen += bsz

    print("\n" + "=" * 120)
    print("[Alpha sweep results]")
    print("alpha = ACT weight. final = alpha * ACT + (1-alpha) * ACM")
    print("=" * 120)
    print(
        f"{'alpha':>6s} | {'L1_to_expert':>12s} | {'diff':>10s} | {'accel':>10s} | "
        f"{'jerk':>10s} | {'signflip':>10s} | {'overlap':>10s}"
    )
    print("-" * 120)

    for alpha in alphas:
        m = mean_dict(metrics_by_alpha[alpha])
        l1 = float(np.mean(l1_by_alpha[alpha]))

        if len(overlap_by_alpha[alpha]) > 0:
            om = mean_dict(overlap_by_alpha[alpha])
            overlap_l2 = om["overlap_l2"]
        else:
            overlap_l2 = float("nan")

        print(
            f"{alpha:6.2f} | "
            f"{l1:12.6f} | "
            f"{m['diff_l2']:10.6f} | "
            f"{m['accel_l2']:10.6f} | "
            f"{m['jerk_l2']:10.6f} | "
            f"{m['sign_flip']:10.6f} | "
            f"{overlap_l2:10.6f}"
        )

    print("=" * 120)
    print(f"Frames used: {n_seen}, overlaps: {n_overlap}")
    print("=" * 120)


if __name__ == "__main__":
    main()
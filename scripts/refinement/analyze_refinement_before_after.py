#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


def smoothness_metrics(actions: torch.Tensor):
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


def make_batch(policy, batch, device):
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    if policy.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in policy.config.image_features]

    return batch


@torch.no_grad()
def forward_before_after(policy, batch):
    """
    Returns:
      raw_actions:     action_head(decoder_out before refinement)
      refined_actions: action_head(refinement(decoder_out))
    """
    model = policy.model
    config = policy.config

    if config.use_vae and ACTION in batch and model.training:
        raise RuntimeError("This analysis should run in eval mode.")

    batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

    latent_sample = torch.zeros(
        [batch_size, config.latent_dim],
        dtype=torch.float32,
        device=batch[OBS_STATE].device,
    )

    encoder_in_tokens = [model.encoder_latent_input_proj(latent_sample)]
    encoder_in_pos_embed = list(model.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

    if config.robot_state_feature:
        encoder_in_tokens.append(model.encoder_robot_state_input_proj(batch[OBS_STATE]))

    if config.env_state_feature:
        encoder_in_tokens.append(model.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

    if config.image_features:
        for img in batch[OBS_IMAGES]:
            cam_features = model.backbone(img)["feature_map"]
            cam_pos_embed = model.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
            cam_features = model.encoder_img_feat_input_proj(cam_features)

            cam_features = torch.einsum("bchw->hwbc", cam_features).reshape(-1, cam_features.shape[0], cam_features.shape[1])
            cam_pos_embed = torch.einsum("bchw->hwbc", cam_pos_embed).reshape(-1, cam_pos_embed.shape[0], cam_pos_embed.shape[1])

            encoder_in_tokens.extend(list(cam_features))
            encoder_in_pos_embed.extend(list(cam_pos_embed))

    encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
    encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

    encoder_out = model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

    decoder_in = torch.zeros(
        (config.chunk_size, batch_size, config.dim_model),
        dtype=encoder_in_pos_embed.dtype,
        device=encoder_in_pos_embed.device,
    )

    decoder_out = model.decoder(
        decoder_in,
        encoder_out,
        encoder_pos_embed=encoder_in_pos_embed,
        decoder_pos_embed=model.decoder_pos_embed.weight.unsqueeze(1),
    )

    decoder_out = decoder_out.transpose(0, 1)  # (B,T,C)

    raw_actions = model.action_head(decoder_out)

    if model.refinement is not None:
        refined_decoder_out = model.refinement(decoder_out)
        refined_actions = model.action_head(refined_decoder_out)
    else:
        refined_actions = raw_actions

    return raw_actions, refined_actions


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

    print(f"[HAS REFINEMENT] {policy.model.refinement is not None}")

    print(f"[LOAD DATASET] {args.dataset_repo_id}")
    dataset = LeRobotDataset(args.dataset_repo_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    raw_chunks = []
    refined_chunks = []

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader)):
            if args.max_batches > 0 and bi >= args.max_batches:
                break

            batch = make_batch(policy, batch, args.device)
            raw, refined = forward_before_after(policy, batch)

            raw_chunks.append(raw.detach().cpu())
            refined_chunks.append(refined.detach().cpu())

    raw_chunks = torch.cat(raw_chunks, dim=0)
    refined_chunks = torch.cat(refined_chunks, dim=0)

    print(f"[RAW CHUNKS]     {raw_chunks.shape}")
    print(f"[REFINED CHUNKS] {refined_chunks.shape}")

    raw_metrics = smoothness_metrics(raw_chunks)
    refined_metrics = smoothness_metrics(refined_chunks)

    print_metrics("Before refinement: action_head(decoder_out)", raw_metrics)
    print_metrics("After refinement: action_head(refinement(decoder_out))", refined_metrics)

    print("\n[After / Before ratio]")
    for k in ["diff", "accel", "jerk"]:
        r = refined_metrics[k]["l2_mean"] / raw_metrics[k]["l2_mean"]
        print(f"{k:>6s} l2_mean ratio = {r:.4f}")


if __name__ == "__main__":
    main()

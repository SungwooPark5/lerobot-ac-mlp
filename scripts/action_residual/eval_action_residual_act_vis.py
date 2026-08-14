"""Stage 3: Evaluate TE-fixed visual action residual network on ACT.

Modes:
  pure_te   : fixed TE only
  residual  : fixed TE + alpha * visual action residual
"""

import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.envs.configs import AlohaEnv


DEFAULT_BASELINE_DIR = Path("/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model")


class VisualActionResidualNetwork(nn.Module):
    def __init__(
        self,
        L,
        state_dim,
        action_dim,
        vis_dim,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_delta=1.0,
        dropout=0.1,
    ):
        super().__init__()
        self.L = L
        self.max_delta = max_delta

        self.state_proj = nn.Linear(state_dim, d_model)
        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, d_model),
            nn.Dropout(p=dropout),
        )

        self.query = nn.Embedding(1, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, state, image_feat, contribs):
        B = state.shape[0]

        state_tok = self.state_proj(state).unsqueeze(1)
        image_tok = self.image_proj(image_feat).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)

        memory = torch.cat([state_tok, image_tok, chunk_toks], dim=1)

        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory).squeeze(1)

        raw_delta = self.head(x)
        delta_action = torch.tanh(raw_delta) * self.max_delta
        return delta_action


def make_te_weights(L, te_coeff, device):
    # runtime convention: contribs[:, 0] = freshest
    w = torch.exp(-te_coeff * torch.arange(L, device=device, dtype=torch.float32))
    w = w / w.sum()
    return w


class TEVisualResidualPolicy:
    def __init__(
        self,
        baseline_policy,
        residual_net,
        L,
        te_coeff=0.01,
        alpha_residual=0.1,
        mode="residual",
        backbone=None,
        image_key="observation.images.top",
        debug=True,
    ):
        self.baseline = baseline_policy
        self.residual_net = residual_net
        self.L = L
        self.te_coeff = te_coeff
        self.alpha_residual = alpha_residual
        self.mode = mode
        self.config = baseline_policy.config
        self.past_chunks = deque(maxlen=L)

        self.backbone = backbone if backbone is not None else baseline_policy.model.backbone
        self.backbone.eval()
        self.image_key = image_key

        self.latest_feat_map = None
        self.debug = debug
        self.debug_done = False

        self.te_weights_full = make_te_weights(L, te_coeff, next(baseline_policy.parameters()).device)

        def _save_backbone_output(module, inputs, output):
            self.latest_feat_map = output

        self.hook_handle = self.backbone.register_forward_hook(_save_backbone_output)

        print("\n[TE DEBUG]")
        print("Runtime contrib convention: contribs[:, 0] = freshest")
        print("te_coeff:", te_coeff)
        print("te_weights first 5:", self.te_weights_full[:5].detach().cpu().numpy())
        print("te_weights last 5:", self.te_weights_full[-5:].detach().cpu().numpy())
        print("te_weights argmax:", self.te_weights_full.argmax().item())

    def reset(self):
        self.past_chunks.clear()
        self.latest_feat_map = None
        self.debug_done = False

    def eval(self):
        self.baseline.eval()
        if self.residual_net is not None:
            self.residual_net.eval()
        self.backbone.eval()
        return self

    def _extract_vis_feat_from_hook(self):
        out = self.latest_feat_map
        if out is None:
            raise RuntimeError("Backbone hook did not capture any feature map.")

        if isinstance(out, torch.Tensor):
            feat_map = out
        elif isinstance(out, dict):
            if "feature_map" in out:
                feat_map = out["feature_map"]
            else:
                # dict일 경우 마지막 tensor fallback
                tensors = [v for v in out.values() if isinstance(v, torch.Tensor)]
                if len(tensors) == 0:
                    raise RuntimeError(f"Backbone output dict has no tensor. keys={list(out.keys())}")
                feat_map = tensors[-1]
        elif isinstance(out, (list, tuple)):
            feat_map = out[-1]
        else:
            raise TypeError(f"Unsupported backbone output type: {type(out)}")

        if feat_map.ndim == 4:
            vis_feat = feat_map.mean(dim=(-2, -1))
        elif feat_map.ndim == 3:
            vis_feat = feat_map.mean(dim=1)
        elif feat_map.ndim == 2:
            vis_feat = feat_map
        else:
            raise ValueError(f"Unexpected feat_map shape: {feat_map.shape}")

        return vis_feat

    @torch.no_grad()
    def select_action(self, batch):
        # predict_action_chunk 안에서 backbone forward가 돌고 hook이 vis feature를 잡음
        chunk = self.baseline.predict_action_chunk(batch)

        self.past_chunks.append(chunk)
        n = len(self.past_chunks)

        contribs = torch.stack([
            self.past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        # n < L일 때는 residual net 학습 분포와 달라서 pure TE만 사용
        te_weights = make_te_weights(n, self.te_coeff, contribs.device)
        a_te = (te_weights.view(1, n, 1) * contribs).sum(dim=1)

        if self.mode == "pure_te" or n < self.L:
            return a_te

        if self.mode != "residual":
            raise ValueError(f"Unknown mode: {self.mode}")

        state = batch["observation.state"]
        vis_feat = self._extract_vis_feat_from_hook()

        delta_a = self.residual_net(state, vis_feat, contribs)
        action = a_te + self.alpha_residual * delta_a

        if self.debug and not self.debug_done:
            print("\n[ACTION RESIDUAL VIS DEBUG]")
            print("n:", n)
            print("state:", tuple(state.shape))
            print("vis_feat:", tuple(vis_feat.shape))
            print("contribs:", tuple(contribs.shape))
            print("alpha_residual:", self.alpha_residual)
            print("a_te abs mean:", a_te.abs().mean().item())
            print("delta_a abs mean:", delta_a.abs().mean().item())
            print("alpha*delta abs mean:", (self.alpha_residual * delta_a).abs().mean().item())
            print("action-a_te abs mean:", (action - a_te).abs().mean().item())
            self.debug_done = True

        return action


def run_eval(args):
    device = torch.device(args.device)

    print(f"Loading ACT baseline from {args.baseline_dir}")
    baseline = ACTPolicy.from_pretrained(str(args.baseline_dir)).to(device)
    baseline.eval()
    config = baseline.config

    residual_net = None
    ckpt = None

    if args.mode == "residual":
        if args.weight_net_path is None:
            raise ValueError("--mode residual requires --weight_net_path")

        print(f"Loading visual action residual network from {args.weight_net_path}")
        ckpt = torch.load(args.weight_net_path, map_location=device)

        L = ckpt["L"]
        te_coeff = ckpt.get("te_coeff", args.te_coeff)
        alpha_residual = args.alpha_residual if args.alpha_residual is not None else ckpt.get("alpha_residual", 0.1)
        image_key = ckpt.get("image_key", "observation.images.top")

        residual_net = VisualActionResidualNetwork(
            L=L,
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            vis_dim=ckpt["vis_dim"],
            d_model=ckpt.get("d_model", 128),
            n_heads=ckpt.get("n_heads", 4),
            n_layers=ckpt.get("n_layers", 2),
            max_delta=ckpt.get("max_delta", 1.0),
            dropout=ckpt.get("dropout", 0.1),
        ).to(device)

        residual_net.load_state_dict(ckpt["model"])
        residual_net.eval()

        print(
            "Loaded TE-fixed visual action residual net: "
            f"L={L}, te_coeff={te_coeff}, alpha_residual={alpha_residual}, "
            f"lambda_residual={ckpt.get('lambda_residual')}, "
            f"max_delta={ckpt.get('max_delta')}, vis_dim={ckpt['vis_dim']}, "
            f"fresh_index={ckpt.get('fresh_index')}"
        )
    else:
        L = args.L
        te_coeff = args.te_coeff
        alpha_residual = args.alpha_residual if args.alpha_residual is not None else 0.1
        image_key = "observation.images.top"
        print(f"Running pure TE wrapper with L={L}")

    policy = TEVisualResidualPolicy(
        baseline_policy=baseline,
        residual_net=residual_net,
        L=L,
        te_coeff=te_coeff,
        alpha_residual=alpha_residual,
        mode=args.mode,
        backbone=baseline.model.backbone,
        image_key=image_key,
        debug=True,
    ).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(args.baseline_dir),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    env_cfg = AlohaEnv(task=args.task)
    env_obj = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)

    while isinstance(env_obj, dict):
        print("make_env returned dict keys:", env_obj.keys())
        env_obj = list(env_obj.values())[0]

    env = env_obj

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    print("\nEval config:")
    print(f"  mode={args.mode}")
    print(f"  task={args.task}")
    print(f"  eval_episodes={args.eval_episodes}")
    print(f"  eval_batch_size={args.eval_batch_size}")
    print(f"  L={L}")
    print(f"  te_coeff={te_coeff}")
    print(f"  alpha_residual={alpha_residual}")

    all_successes = []

    for start in range(0, args.eval_episodes, args.eval_batch_size):
        end = min(start + args.eval_batch_size, args.eval_episodes)
        batch_n = end - start
        seeds = list(range(start, end))

        if batch_n != args.eval_batch_size:
            # make_env가 고정 n_envs라 마지막 batch는 되도록 eval_episodes를 batch_size 배수로 사용 추천
            print(f"Skipping partial batch {start}:{end}. Use eval_episodes multiple of eval_batch_size.")
            break

        print(f"\nRunning ACT rollout batch: seeds {start}..{end - 1}, batch={batch_n}")

        obs_dict, _ = env.reset(seed=seeds)

        done = np.zeros(batch_n, dtype=bool)
        successes = np.zeros(batch_n, dtype=bool)
        max_steps = env.call("_max_episode_steps")[0]

        policy.reset()

        step = 0
        while not np.all(done) and step < max_steps:
            obs = preprocess_observation(obs_dict)
            obs = add_envs_task(env, obs)
            obs = env_preprocessor(obs)
            obs = preprocessor(obs)

            with torch.inference_mode():
                action = policy.select_action(obs)

            action = postprocessor(action)
            action = env_postprocessor({"action": action})["action"]
            action_np = action.to("cpu").numpy()

            obs_dict, reward, terminated, truncated, info = env.step(action_np)
            done = done | terminated | truncated

            if "final_info" in info:
                final_info = info["final_info"]

                if isinstance(final_info, dict):
                    if "is_success" in final_info:
                        successes |= np.array(final_info["is_success"], dtype=bool)
                else:
                    for i, fi in enumerate(final_info):
                        if isinstance(fi, dict) and fi.get("is_success", False):
                            successes[i] = True

            step += 1
            if step % 50 == 0:
                print(f"step {step}/{max_steps}, done {done.sum()}/{batch_n}")

        all_successes.extend(successes.tolist())

    env.close()

    all_successes = np.array(all_successes, dtype=bool)
    pc_success = all_successes.mean() if len(all_successes) > 0 else 0.0

    print("\nResults ACT TE-fixed visual action residual eval")
    print(f"mode = {args.mode}")
    print(f"pc_success = {pc_success:.3f} ({all_successes.sum()}/{len(all_successes)})")

    print("\nReference:")
    print("  ACT baseline n=50 commit:        62%")
    print("  LeRobot TE fresh-favored:        54%")
    print("  our wrapper pure TE no network:  58~63%")
    print("  action residual no image:        64~67%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="residual", choices=["pure_te", "residual"])
    parser.add_argument("--baseline_dir", type=str, default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--weight_net_path", type=str, default=None)

    parser.add_argument("--task", type=str, default="AlohaTransferCube-v0")
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=50)

    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--te_coeff", type=float, default=0.01)
    parser.add_argument("--alpha_residual", type=float, default=None)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()

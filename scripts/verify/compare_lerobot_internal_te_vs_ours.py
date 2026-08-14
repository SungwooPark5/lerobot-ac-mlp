import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import argparse
from collections import deque
from pathlib import Path
import shutil
import json
import tempfile

import torch
import torch.nn.functional as F
import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors


DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"


def import_policy(policy_type):
    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy
    elif policy_type == "acm":
        from lerobot.policies.acm.modeling_acm import ACMPolicy
        return ACMPolicy
    else:
        raise ValueError(f"unknown policy_type={policy_type}")

def make_temp_policy_dir_with_coeff(src_dir, coeff, n_action_steps=1):
    """
    from_pretrained()가 config.json을 직접 다시 읽기 때문에,
    내부 LeRobot TE를 켜려면 임시 pretrained_model 폴더의 config.json을 수정해서 로드해야 함.
    """
    src_dir = Path(src_dir)
    tmp_root = Path(tempfile.mkdtemp(prefix="tmp_lerobot_te_"))
    dst_dir = tmp_root / "pretrained_model"

    shutil.copytree(src_dir, dst_dir)

    cfg_path = dst_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())

    if "temporal_ensemble_coeff" in cfg:
        cfg["temporal_ensemble_coeff"] = coeff
    else:
        print("[TEMP CONFIG WARN] temporal_ensemble_coeff not in config; adding may fail if config schema rejects it")
        cfg["temporal_ensemble_coeff"] = coeff

    if "n_action_steps" in cfg:
        cfg["n_action_steps"] = n_action_steps

    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[TEMP POLICY DIR] {dst_dir}")
    print(f"[TEMP CONFIG] temporal_ensemble_coeff={coeff}, n_action_steps={cfg.get('n_action_steps')}")
    return dst_dir


def reset_policy(policy):
    # ACTPolicy.reset()이 temporal_ensembler가 없는데도 호출하는 경우가 있어서 방어
    if hasattr(policy, "temporal_ensembler"):
        try:
            policy.reset()
            print("[RESET] called policy.reset()")
            return
        except Exception as e:
            print(f"[RESET WARN] policy.reset() failed: {e}")

    for name in ["reset_buffers", "reset_queues"]:
        if hasattr(policy, name):
            try:
                getattr(policy, name)()
                print(f"[RESET] called policy.{name}()")
                return
            except TypeError:
                pass

    print("[RESET] skipped")


def set_coeff_everywhere(obj, coeff):
    """
    config/instance 내부에 temporal_ensemble_coeff가 있으면 최대한 강제로 설정.
    """
    count = 0

    if hasattr(obj, "temporal_ensemble_coeff"):
        setattr(obj, "temporal_ensemble_coeff", coeff)
        count += 1

    if hasattr(obj, "config") and hasattr(obj.config, "temporal_ensemble_coeff"):
        setattr(obj.config, "temporal_ensemble_coeff", coeff)
        count += 1

    if hasattr(obj, "model") and hasattr(obj.model, "temporal_ensemble_coeff"):
        setattr(obj.model, "temporal_ensemble_coeff", coeff)
        count += 1

    if hasattr(obj, "model") and hasattr(obj.model, "config") and hasattr(obj.model.config, "temporal_ensemble_coeff"):
        setattr(obj.model.config, "temporal_ensemble_coeff", coeff)
        count += 1

    return count


def maybe_force_n_action_steps_one(config):
    """
    TE 비교는 매 step select_action 해야 하므로 n_action_steps=1 강제 시도.
    없는 필드는 건드리지 않음.
    """
    for attr in ["n_action_steps"]:
        if hasattr(config, attr):
            print(f"[CONFIG] {attr}: {getattr(config, attr)} -> 1")
            setattr(config, attr, 1)


def make_batch(sample, preprocessor):
    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v
    return preprocessor(batch)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_type", choices=["act", "acm"], required=True)
    parser.add_argument("--baseline_dir", type=str, required=True)

    # 네가 실제 비교하던 대응:
    # our pos(+0.01) ↔ lerobot neg(-0.01)
    # our neg(-0.01) ↔ lerobot pos(+0.01)
    parser.add_argument("--lerobot_coeff", type=float, required=True)
    parser.add_argument("--our_coeff", type=float, required=True)

    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=160)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    PolicyCls = import_policy(args.policy_type)
    baseline_dir = Path(args.baseline_dir)

    print("=" * 100)
    print("Compare LeRobot internal TE select_action() vs OUR wrapper contribs")
    print(f"policy_type     = {args.policy_type}")
    print(f"baseline_dir    = {baseline_dir}")
    print(f"lerobot_coeff   = {args.lerobot_coeff}")
    print(f"our_coeff       = {args.our_coeff}")
    print(f"L               = {args.L}")
    print(f"frames          = {args.start_idx} ~ {args.start_idx + args.num_frames - 1}")
    print(f"device          = {device}")
    print("=" * 100)

    # ------------------------------------------------------------
    # 1) LeRobot internal TE policy
    # ------------------------------------------------------------
    # from_pretrained()가 baseline_dir/config.json을 다시 읽으므로,
    # 임시 pretrained_model 폴더를 만들어 config.json 자체에 coeff를 반영한다.
    lr_baseline_dir = make_temp_policy_dir_with_coeff(
        baseline_dir,
        coeff=args.lerobot_coeff,
        n_action_steps=1,
    )

    policy_lr = PolicyCls.from_pretrained(str(lr_baseline_dir)).to(device).eval()
    print(f"[POLICY_LR] has temporal_ensembler = {hasattr(policy_lr, 'temporal_ensembler')}")
    nset = set_coeff_everywhere(policy_lr, args.lerobot_coeff)
    print(f"[POLICY_LR] set coeff count = {nset}")
    reset_policy(policy_lr)

    # ------------------------------------------------------------
    # 2) Chunk-only policy for OUR contribs
    # ------------------------------------------------------------
    # 우리 contribs는 predict_action_chunk만 쓰므로 원본 baseline_dir로 로드한다.
    policy_ours = PolicyCls.from_pretrained(str(baseline_dir)).to(device).eval()
    reset_policy(policy_ours)

    # preprocessors
    preprocessor_lr, postprocessor_lr = make_pre_post_processors(
        policy_cfg=policy_lr.config,
        pretrained_path=str(lr_baseline_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    preprocessor_ours, postprocessor_ours = make_pre_post_processors(
        policy_cfg=policy_ours.config,
        pretrained_path=str(baseline_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # dataset
    dataset = LeRobotDataset(DATASET_REPO_ID)
    print(f"[DATASET] len={len(dataset)}, episodes={dataset.num_episodes}")

    past_chunks = deque(maxlen=args.L)

    rows = []
    diffs = []
    diffs_rev_weight = []

    print("\nstep, n, diff_lerobot_vs_ours, diff_lerobot_vs_ours_reversed_weight, our_argmax, reversed_argmax")
    print("-" * 100)

    for local_t in range(args.num_frames):
        idx = args.start_idx + local_t
        sample = dataset[idx]

        batch_lr = make_batch(sample, preprocessor_lr)
        batch_ours = make_batch(sample, preprocessor_ours)

        # LeRobot internal TE action
        action_lr = policy_lr.select_action(batch_lr)
        action_lr = postprocessor_lr(action_lr)

        # OUR chunk candidates
        chunk = policy_ours.predict_action_chunk(batch_ours)
        past_chunks.append(chunk)

        n = len(past_chunks)

        contribs = torch.stack([
            past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        # OUR intended prior: idx=0 latest 기준 coeff * arange
        logits_ours = args.our_coeff * torch.arange(n, device=device, dtype=contribs.dtype)
        w_ours = torch.softmax(logits_ours, dim=0)

        action_ours = (w_ours.view(1, n, 1) * contribs).sum(dim=1)
        action_ours = postprocessor_ours(action_ours)

        # 비교용: 같은 contribs에 weight만 reverse해서도 비교
        w_rev = torch.flip(w_ours, dims=[0])
        action_ours_rev_weight = (w_rev.view(1, n, 1) * contribs).sum(dim=1)
        action_ours_rev_weight = postprocessor_ours(action_ours_rev_weight)

        diff = (action_lr - action_ours).abs().max().item()
        diff_rev = (action_lr - action_ours_rev_weight).abs().max().item()

        diffs.append(diff)
        diffs_rev_weight.append(diff_rev)

        if local_t < 10 or local_t in [args.L - 2, args.L - 1, args.L, args.L + 1, args.num_frames - 1]:
            print(
                f"{local_t:4d}, {n:3d}, "
                f"{diff:.10f}, {diff_rev:.10f}, "
                f"{int(w_ours.argmax().item())}, {int(w_rev.argmax().item())}"
            )

        rows.append({
            "local_t": local_t,
            "dataset_idx": idx,
            "n": n,
            "diff_lerobot_vs_ours": diff,
            "diff_lerobot_vs_ours_reversed_weight": diff_rev,
            "our_argmax": int(w_ours.argmax().item()),
            "reversed_argmax": int(w_rev.argmax().item()),
            "our_weight_first": float(w_ours[0].item()),
            "our_weight_last": float(w_ours[-1].item()),
            "reversed_weight_first": float(w_rev[0].item()),
            "reversed_weight_last": float(w_rev[-1].item()),
        })

    diffs = np.asarray(diffs)
    diffs_rev_weight = np.asarray(diffs_rev_weight)

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    print(f"mean diff LeRobot vs OUR                 = {diffs.mean():.10f}")
    print(f"max  diff LeRobot vs OUR                 = {diffs.max():.10f}")
    print(f"mean diff LeRobot vs OUR reversed weight = {diffs_rev_weight.mean():.10f}")
    print(f"max  diff LeRobot vs OUR reversed weight = {diffs_rev_weight.max():.10f}")

    if diffs.mean() < diffs_rev_weight.mean():
        print("\n=> LeRobot internal TE가 OUR contribs + OUR weight와 더 가깝다.")
        print("   즉 후보 순서/weight 방향이 대체로 맞을 가능성이 큼.")
    else:
        print("\n=> LeRobot internal TE가 OUR contribs + REVERSED weight와 더 가깝다.")
        print("   즉 후보 순서 또는 weight 적용 방향이 뒤집혀 있을 가능성이 큼.")

    out_csv = Path(
        f"/home1/eunji24/lerobot_project/outputs/compare_lerobot_internal_te_vs_ours_"
        f"{args.policy_type}_lr{args.lerobot_coeff}_our{args.our_coeff}.csv".replace("-", "neg").replace(".", "p")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import csv
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SAVED] {out_csv}")


if __name__ == "__main__":
    main()

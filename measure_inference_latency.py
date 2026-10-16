#!/usr/bin/env python
"""Single-column inference latency measurement for plain / dual_forward / action_only,
to drop into the K=100 comparison table alongside success rate / param count.

Times end-to-end preprocessor + `predict_action_chunk()` (the same call path used at
inference in `select_action`) on one fixed, real observation batch pulled once from the
dataset -- the same batch is reused for every model and every timed iteration, so only
the model/preprocessor differs across rows. GPU timing uses `torch.cuda.synchronize()`
around each iteration so async kernel launches don't make the numbers optimistic.

Usage:
    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-final/src:/home1/eunji24/lerobot_project/lerobot-final
    /home1/eunji24/venvs/lerobot_libero/bin/python measure_inference_latency.py \
        --ckpt act=/home1/eunji24/lerobot_project/outputs/fair_ksweep/libero_10/act_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt plain=/home1/eunji24/lerobot_project/outputs/fair_ksweep/libero_10/plain_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt dual_forward=/home1/eunji24/lerobot_project/outputs/nopairs/libero_10/scan_same_K100/seed0/checkpoints/150000/pretrained_model \
        --ckpt action_only=/home1/eunji24/lerobot_project/outputs/nopairs/libero_10/bimamba_actiononly_K100/seed0/checkpoints/150000/pretrained_model \
        --batch-size 1 --n-warmup 10 --n-iters 100

--batch-size 1 matches real single-robot deployment latency; pass --batch-size 5 to
match this project's eval.batch_size=5 if you want a throughput-style number instead.
"""

import argparse
import time

import numpy as np
import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

# Rows are printed in whatever order --ckpt args are given; any names are accepted.


def parse_ckpt_arg(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--ckpt must be name=path, got: {spec!r}")
    name, path = spec.split("=", 1)
    return name, path


def load_policy(ckpt_dir: str, dataset, device: str):
    cfg = TrainPipelineConfig.from_pretrained(ckpt_dir)
    cfg.policy.pretrained_path = str(ckpt_dir)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.to(device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=str(ckpt_dir))
    return policy, preprocessor


@torch.no_grad()
def time_inference(policy, preprocessor, raw_batch, device: str, n_warmup: int, n_iters: int):
    for _ in range(n_warmup):
        pb = preprocessor(raw_batch)
        _ = policy.predict_action_chunk(pb)
    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = np.empty(n_iters)
    for i in range(n_iters):
        t0 = time.perf_counter()
        pb = preprocessor(raw_batch)
        _ = policy.predict_action_chunk(pb)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms[i] = (time.perf_counter() - t0) * 1000.0
    return times_ms.mean(), times_ms.std(), np.median(times_ms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True, type=parse_ckpt_arg, dest="ckpts",
                    metavar="name=path", help="repeatable: name=pretrained_model_dir. Any names -- pass whichever checkpoints you want in the table.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--n-warmup", type=int, default=10)
    ap.add_argument("--n-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpts = dict(args.ckpts)
    names = [n for n, _ in args.ckpts]
    if len(set(names)) != len(names):
        raise SystemExit(f"--ckpt names must be unique, got: {names}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  batch_size={args.batch_size}  n_warmup={args.n_warmup}  n_iters={args.n_iters}")

    base_cfg = TrainPipelineConfig.from_pretrained(ckpts[names[0]])
    print(f"loading dataset {base_cfg.dataset.repo_id} to pull one real observation batch ...")
    dataset = make_dataset(base_cfg)

    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(dataset), generator=g)[: args.batch_size].tolist()
    subset = torch.utils.data.Subset(dataset, idx)
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    raw_batch = next(iter(loader))
    raw_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in raw_batch.items()}

    print(f"\n{'model':<14}{'mean_ms':>10}{'std_ms':>10}{'median_ms':>12}")
    for name in names:
        path = ckpts[name]
        print(f"[{name}] loading {path}")
        policy, preprocessor = load_policy(path, dataset, device)
        mean, std, median = time_inference(policy, preprocessor, raw_batch, device, args.n_warmup, args.n_iters)
        print(f"{name:<14}{mean:>10.2f}{std:>10.2f}{median:>12.2f}")
        del policy, preprocessor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

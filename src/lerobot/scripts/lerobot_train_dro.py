#!/usr/bin/env python
"""lerobot_train_dro.py — standard lerobot_train with DRO envs registered.

Use with --env.task=Aloha*DRO-v0 to get the SPLIT training-time eval: each eval
budget runs a clean half and a disturbed half on the same env instances / same seeds
(paired). Disturbance params come from env vars (typically DRO_TYPE=mix DRO_LEVEL=2
DRO_STEP=-1 — per-episode random type and timing); see lerobot/envs/dro_perturb.py.

    CUDA_VISIBLE_DEVICES=0 DRO_TYPE=mix DRO_LEVEL=2 DRO_STEP=-1 \\
    python -m lerobot.scripts.lerobot_train_dro \\
        --policy.type=acm2_dro --env.type=aloha --env.task=AlohaTransferCubeDRO-v0 \\
        --eval_freq=10000 --eval_start_step=50000 --eval.n_episodes=400 ...

Logged per eval step: pc_success (clean 200ep) + dro_pc_success (disturbed 200ep)
in training_log.jsonl; eval_curve + dro_eval_curve in metrics.json.
"""
import lerobot.envs.dro_perturb  # noqa: F401  ← MUST precede lerobot_train (registers DRO envs)
from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()

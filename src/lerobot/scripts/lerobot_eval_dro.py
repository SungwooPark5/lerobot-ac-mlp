#!/usr/bin/env python
"""lerobot_eval_dro.py — run the standard lerobot_eval on a DRO-disturbed gym_aloha env.

Importing lerobot.envs.dro_perturb registers the disturbed env ids BEFORE lerobot's
make_env runs, so the entire validated lerobot_eval pipeline (correct normalization,
official SR, videos) is reused unchanged — only the env injects a disturbance.
Disturbance params come from env vars (DRO_TYPE, DRO_LEVEL, DRO_STEP, ... —
see lerobot/envs/dro_perturb.py).

    CUDA_VISIBLE_DEVICES=0 DRO_TYPE=push DRO_LEVEL=2 DRO_SEED=0 \\
    DRO_LOG_DIR=<out>/dro_logs RECORD_DIR=<out>/actions \\
    python -m lerobot.scripts.lerobot_eval_dro \\
        --policy.path=<.../pretrained_model> --env.type=aloha \\
        --env.task=AlohaTransferCubeDRO-v0 --eval.n_episodes=500 \\
        --eval.batch_size=50 --output_dir=<out>

DRO_TYPE=none (default) runs the clean baseline through the identical wrapper stack.
"""
import lerobot.envs.dro_perturb  # noqa: F401  ← MUST precede lerobot_eval (registers DRO envs)
from lerobot.scripts.lerobot_eval import main

if __name__ == "__main__":
    main()

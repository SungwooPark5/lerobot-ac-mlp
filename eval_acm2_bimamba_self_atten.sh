#!/bin/bash
# val/test eval for acm2_bimamba_self_atten (bimamba on, self-atten off) -- lerobot-ac-mlp worktree.
# Re-evals 5 checkpoints (50 ep) per seed to pick best-val, then held-out test (seed 100000, 500 ep),
# prints test mean +/- std. Run AFTER training. Passes flags through, e.g.: bash eval_...sh --seeds 3 4
set -e
export MUJOCO_GL=egl
export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-ac-mlp/src:/home1/eunji24/lerobot_project/lerobot-ac-mlp
cd "$(dirname "$0")"
python eval_acm2_bimamba_self_atten.py "$@"

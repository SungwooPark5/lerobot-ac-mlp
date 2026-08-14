#!/bin/bash

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$HOME/lerobot_project/lerobot-ac-mlp/src:$HOME/lerobot_project/lerobot-ac-mlp:$PYTHONPATH

ENV="aloha"
TASK="AlohaTransferCube-v0"
DATASET="lerobot/aloha_sim_transfer_cube_human"

INIT_CKPT="$HOME/lerobot_project/outputs/train/ensemble_network_init_100k_L50"
RUN_NAME="act_network2_L50_ensemble_network_freeze_from100k_50k_b2"
ACT_OUTPUT="$HOME/lerobot_project/outputs/train/$RUN_NAME"

echo "============================================================"
echo "  Ensemble Network: freeze action + train weight head only"
echo "  baseline: 100k"
echo "  L:        50"
echo "  batch:    4"
echo "  Init:     $INIT_CKPT"
echo "  Output:   $ACT_OUTPUT"
echo "============================================================"

python -m lerobot.scripts.lerobot_train \
  --policy.path=$INIT_CKPT \
  --policy.optimizer_lr=1e-5 \
  --env.type=$ENV \
  --env.task=$TASK \
  --dataset.repo_id=$DATASET \
  --dataset.video_backend=pyav \
  --steps=50000 \
  --eval_freq=10000 \
  --save_freq=50000 \
  --batch_size=2 \
  --eval.n_episodes=50 \
  --eval.batch_size=50 \
  --output_dir=$ACT_OUTPUT \
  --wandb.enable=true \
  --policy.push_to_hub=false
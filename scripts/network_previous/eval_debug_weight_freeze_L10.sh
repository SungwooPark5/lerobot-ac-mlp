#!/bin/bash

export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-ac-mlp/src:/home1/eunji24/lerobot_project/lerobot-ac-mlp:$PYTHONPATH

cd /home1/eunji24/lerobot_project/lerobot-ac-mlp

POLICY_PATH="/home1/eunji24/lerobot_project/outputs/train/act_network2_L10_ensemble_network_freeze_100k/checkpoints/020000/pretrained_model"

echo "============================================================"
echo " Eval debug: L10 freeze checkpoint"
echo " Policy path: $POLICY_PATH"
echo "============================================================"

ls -lh "$POLICY_PATH"

python src/lerobot/scripts/lerobot_eval.py \
  --policy.path="$POLICY_PATH" \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=1 \
  --eval.batch_size=1

import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

BASELINE_DIR = Path("/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_transfer_cube_test_100k/checkpoints/060000/pretrained_model")
DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
CACHE_DIR = Path("/home1/eunji24/lerobot_project/cache/acm_baseline_chunks")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ACM baseline from {BASELINE_DIR}")
config = PreTrainedConfig.from_pretrained(str(BASELINE_DIR))
print(f"policy_type={type(config).__name__}, chunk_size={config.chunk_size}")

policy = ACMPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
policy.eval()

preprocessor, _ = make_pre_post_processors(
    policy_cfg=config,
    pretrained_path=str(BASELINE_DIR),
    preprocessor_overrides={
        "device_processor": {"device": str(DEVICE)},
    },
)

print(f"Loading dataset {DATASET_REPO_ID}")
dataset = LeRobotDataset(DATASET_REPO_ID)
N = len(dataset)
print(f"Total frames={N}, Episodes={dataset.num_episodes}")

sample0 = dataset[0]
state_dim = sample0["observation.state"].shape[0]
action_dim = sample0["action"].shape[0]
print(f"state_dim={state_dim}, action_dim={action_dim}")

chunks_arr = np.zeros((N, config.chunk_size, action_dim), dtype=np.float32)
states_arr = np.zeros((N, state_dim), dtype=np.float32)
actions_arr = np.zeros((N, action_dim), dtype=np.float32)
ep_idx_arr = np.zeros(N, dtype=np.int64)

# Episode 경계                                                                                                                                                                                      
ep_starts = []  
for ep in range(dataset.num_episodes):
    episode_dict = dataset.meta.episodes[ep]
    f = int(episode_dict["dataset_from_index"])
    t = int(episode_dict["dataset_to_index"])
    ep_starts.append((f, t))
                                                                                                                                                                             # Episode index O(N)으로 채우기
for ep, (f, t) in enumerate(ep_starts):
    ep_idx_arr[f:t] = ep

print("Computing ACM chunks...")
with torch.no_grad():
    for idx in tqdm(range(N)):
        sample = dataset[idx]
        batch = {}
        for k, v in sample.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = v

        batch = preprocessor(batch)
        chunk = policy.predict_action_chunk(batch)

        chunks_arr[idx] = chunk[0].detach().cpu().numpy()
        states_arr[idx] = sample["observation.state"].numpy()
        actions_arr[idx] = sample["action"].numpy()

print("Saving cache...")
np.savez_compressed(
    CACHE_DIR / "chunks.npz",
    chunks=chunks_arr,
    states=states_arr,
    actions=actions_arr,
    episode_idx=ep_idx_arr,
    ep_starts=np.array(ep_starts),
)

print(f"Saved to {CACHE_DIR / 'chunks.npz'}")
print(f"chunks: {chunks_arr.shape}, ~{chunks_arr.nbytes/1e6:.1f}MB")
print(f"states: {states_arr.shape}, actions: {actions_arr.shape}")
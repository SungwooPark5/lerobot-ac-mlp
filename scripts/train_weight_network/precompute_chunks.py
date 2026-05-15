"""Stage 1: baseline ACT로 dataset 모든 frame chunk 미리 계산."""                                                                                                  
import sys                                                                                                                                                         
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")                                                                                            
                                          
import torch                                                                                                                                                       
import numpy as np
from pathlib import Path                                                                                                                                           
from tqdm import tqdm
                                                                                                                                                                 
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset                                                                                                        
from lerobot.policies.factory import make_pre_post_processors
                                                                                                                                                                 
BASELINE_DIR = Path("/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model")
DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"                                                                                                          
CACHE_DIR = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks")
CACHE_DIR.mkdir(parents=True, exist_ok=True)                                                                                                                       
                                      
# 1. baseline policy 로드                                                                                                                                          
print(f"Loading baseline from {BASELINE_DIR}")                                                                                                                     
config = PreTrainedConfig.from_pretrained(str(BASELINE_DIR))                                                                                                       
print(f"  use_vae: {config.use_vae}, chunk_size: {config.chunk_size}")                                                                                             
policy = ACTPolicy.from_pretrained(str(BASELINE_DIR)).cuda()
policy.eval()                                                                                                                                                      

# 2. preprocessor                                                                                                                                                  
preprocessor, _ = make_pre_post_processors(
  policy_cfg=config,                                                                                                                                             
  pretrained_path=str(BASELINE_DIR),  
  preprocessor_overrides={
      "device_processor": {"device": "cuda"},                                                                                                                    
  },                                      
)                                                                                                                                                                  
              
# 3. dataset                                                                                                                                                       
print(f"Loading dataset {DATASET_REPO_ID}")
dataset = LeRobotDataset(DATASET_REPO_ID)                                                                                                                          
N = len(dataset)                            
print(f"  Total frames: {N}, Episodes: {dataset.num_episodes}")
                                          
# 4. shapes 결정                                                                                                                                                   
sample0 = dataset[0]
state_dim = sample0["observation.state"].shape[0]                                                                                                                  
action_dim = sample0["action"].shape[0]     
print(f"  state_dim={state_dim}, action_dim={action_dim}")                                                                                                         
                                          
# 5. 저장 버퍼                                                                                                                                                     
chunks_arr = np.zeros((N, config.chunk_size, action_dim), dtype=np.float32)
states_arr = np.zeros((N, state_dim), dtype=np.float32)                                                                                                            
actions_arr = np.zeros((N, action_dim), dtype=np.float32)
ep_idx_arr = np.zeros(N, dtype=np.int64)                                                                                                                           
                                                                                                                                                                 
# Episode 경계 계산
ep_starts = []
frames_per_ep = N // dataset.num_episodes

for ep in range(dataset.num_episodes):
    f = ep * frames_per_ep
    t = (ep + 1) * frames_per_ep
    ep_starts.append((f, t))

print(f"  frames_per_episode={frames_per_ep}")                                                                                                                                    
                                      
# 6. iterate                                                                                                                                                       
print("Computing chunks...")                                                                                                                                       
with torch.no_grad():
  for idx in tqdm(range(N)):                                                                                                                                     
      sample = dataset[idx]               
      # batchify                      
      batch = {}
      for k, v in sample.items():                                                                                                                                
          if torch.is_tensor(v):
              batch[k] = v.unsqueeze(0)                                                                                                                          
          else:                       
              batch[k] = v
      # preprocess                                                                                                                                               
      batch = preprocessor(batch)
      # forward                                                                                                                                                  
      chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
      chunks_arr[idx] = chunk[0].cpu().numpy()
      states_arr[idx] = sample["observation.state"].numpy()
      actions_arr[idx] = sample["action"].numpy()
      # episode index 한 번에 채우기
      ep_idx_arr = np.zeros(N, dtype=np.int64)
      for ep, (f, t) in enumerate(ep_starts):
          ep_idx_arr[f:t] = ep                                                                                                                                             
                                                                                                                                                                 
# 7. save
print("Saving cache...")                                                                                                                                           
np.savez_compressed(
  CACHE_DIR / "chunks.npz",               
  chunks=chunks_arr,                  
  states=states_arr,
  actions=actions_arr,                                                                                                                                           
  episode_idx=ep_idx_arr,
  ep_starts=np.array(ep_starts),                                                                                                                                 
)                                       
print(f"\nSaved to {CACHE_DIR / 'chunks.npz'}")
print(f"  chunks: {chunks_arr.shape}")                                                                                                                             
print(f"  states: {states_arr.shape}")  
print(f"  actions: {actions_arr.shape}")                                                                                                                           
print(f"  size: ~{chunks_arr.nbytes / 1e6:.1f} MB chunks + {states_arr.nbytes/1e6:.1f}MB states")   
"""Stage 2: 캐시된 chunks로 weight network 학습 (메모리 안전 + 벡터화)."""                
import sys                                                                                
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")                   
                                          
import torch                                                                              
import torch.nn as nn
import torch.nn.functional as F                                                           
import numpy as np                      
from pathlib import Path                                                                  
from tqdm import tqdm                                                                     
import wandb
                                                                                        
CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/weight_network_decoder")
OUT_DIR.mkdir(parents=True, exist_ok=True)                                                
                                                                                        
# === 하이퍼파라미터 ===                                                                  
L = 100                                                                                   
D_MODEL = 128                                                                             
N_HEADS = 4                                                                               
N_LAYERS = 2                                                                              
LR = 3e-4                                                                                 
STEPS = 50000   
BATCH_SIZE = 64                                                                           
LOG_FREQ = 100                          
SAVE_FREQ = 10000                                                                         
DEVICE = torch.device("cuda")                                                             
                                      
                                                                                        
class WeightNetworkDecoder(nn.Module):                                                    
  def __init__(self, L, state_dim, action_dim, d_model=128, n_heads=4, n_layers=2):
      super().__init__()                                                                
      self.L = L                      
      self.chunk_proj = nn.Linear(action_dim, d_model)                                  
      self.state_proj = nn.Linear(state_dim, d_model)                                   
      self.queries = nn.Embedding(L, d_model)
      decoder_layer = nn.TransformerDecoderLayer(                                       
          d_model, n_heads,                                                             
          dim_feedforward=4*d_model,
          batch_first=True,                                                             
      )                               
      self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)                     
      self.head = nn.Linear(d_model, 1)                                                 
                                      
  def forward(self, state, contribs):                                                   
      B = state.shape[0]                                                                
      state_tok = self.state_proj(state).unsqueeze(1)
      chunk_toks = self.chunk_proj(contribs)                                            
      memory = torch.cat([state_tok, chunk_toks], dim=1)
      q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)                            
      x = self.decoder(q, memory)                                                       
      return self.head(x).squeeze(-1)     
                                                                                        
                                          
# 1. cache 로드 (CPU에 유지)                                                              
print("Loading cache...")
cache = np.load(CACHE_PATH)                                                               
chunks_cpu = torch.from_numpy(cache["chunks"]).float()      # CPU
states_cpu = torch.from_numpy(cache["states"]).float()      # CPU                         
actions_cpu = torch.from_numpy(cache["actions"]).float()    # CPU
ep_starts = cache["ep_starts"].astype(np.int64)             # numpy                       
                                          
N, chunk_size, action_dim = chunks_cpu.shape                                              
state_dim = states_cpu.shape[1]         
print(f"  N={N}, chunk_size={chunk_size}, state_dim={state_dim}, action_dim={action_dim}")
print(f"  cache size: chunks={chunks_cpu.element_size()*chunks_cpu.nelement()/1e6:.1f}MB")
                                                                                        
# Episode 길이 검증                                                                       
ep_lens = ep_starts[:, 1] - ep_starts[:, 0]                                               
print(
    f"  episode length: min={ep_lens.min()}, "
    f"max={ep_lens.max()}, mean={ep_lens.mean():.0f}"
)                                                              
assert ep_lens.min() >= L, f"Some episodes too short for L={L}!"                          
                                                                                        
                                                                                        
# 2. 벡터화된 batch sampling (numpy indexing → 마지막에 GPU)                              
def sample_batch(batch_size):                                                             
  # Episode 선택 (with valid length)                                                    
  valid_eps = np.where(ep_lens >= L)[0]   
  eps = np.random.choice(valid_eps, size=batch_size, replace=True)                      
                                          
  # 각 episode 내 window 시작 위치                                                      
  win_starts = np.zeros(batch_size, dtype=np.int64)
  for i, ep in enumerate(eps):                                                          
      ep_from = ep_starts[ep, 0]                                                        
      ep_len = ep_lens[ep]                                                              
      offset = np.random.randint(0, ep_len - L + 1)                                     
      win_starts[i] = ep_from + offset                                                  

  # contribs[b, i] = chunks[win_starts[b] + (L-1) - i, i]                               
  # 벡터화 인덱스 생성 (모두 numpy)   
  contribs_frame_idx = win_starts[:, None] + (L - 1) - np.arange(L)[None, :]  # (B, L)  
  pos_idx = np.broadcast_to(np.arange(L), (batch_size, L))                     # (B, L) 
                                          
  # CPU tensor indexing (numpy index 사용 가능)                                         
  contribs = chunks_cpu[contribs_frame_idx, pos_idx]   # (B, L, action_dim)             
                                                                                        
  target_idx = win_starts + L - 1  # (B,)                                               
  states = states_cpu[target_idx]                                                       
  targets = actions_cpu[target_idx]                                                     
                                          
  # GPU로 transfer (한 번에)                                                            
  return states.to(DEVICE), contribs.to(DEVICE), targets.to(DEVICE)
                                                                                        
                                      
# 3. 모델                                                                                 
model = WeightNetworkDecoder(L, state_dim, action_dim, D_MODEL, N_HEADS,                  
N_LAYERS).to(DEVICE)                        
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)                                  
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
                                                                                        
wandb.init(project="lerobot", name=f"weight_decoder_L{L}", config={
  "L": L, "lr": LR, "steps": STEPS, "batch_size": BATCH_SIZE,                           
  "d_model": D_MODEL, "n_layers": N_LAYERS,                                             
})                                                                                        
                                                                                        
# 4. random init baseline 측정
with torch.no_grad():
    s, c, t = sample_batch(256)
    logits = model(s, c)
    w = F.softmax(logits, dim=1).unsqueeze(-1)
    ens = (w * c).sum(dim=1)
    init_loss = F.l1_loss(ens, t).item()
    init_entropy = -(
        F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)
    ).sum(dim=1).mean().item()

    print(
        f"\nInitial random-init: loss={init_loss:.4f}, "
        f"entropy={init_entropy:.3f} "
        f"(uniform={np.log(L):.3f})"
    )

# 5. 학습
print(f"\nTraining {STEPS} steps...")
for step in tqdm(range(STEPS)):
    s, c, t = sample_batch(BATCH_SIZE)

    logits = model(s, c)
    weights = F.softmax(logits, dim=1).unsqueeze(-1)
    ensembled = (weights * c).sum(dim=1)
    loss = F.l1_loss(ensembled, t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % LOG_FREQ == 0:
        with torch.no_grad():
            w_flat = weights[:, :, 0]
            entropy = -(w_flat * (w_flat + 1e-9).log()).sum(dim=1).mean()
            wandb.log({
                "loss": loss.item(),
                "step": step,
                "weight_entropy": entropy.item(),
                "weight_entropy_ratio": entropy.item() / np.log(L),
                "weight_max": w_flat.max(dim=1).values.mean().item(),
                "weight_logits_std_per_step": logits.std(dim=1).mean().item(),
            })

    if (step + 1) % SAVE_FREQ == 0:
        torch.save({
            "model": model.state_dict(),
            "step": step + 1,
            "L": L,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
        }, OUT_DIR / f"weight_net_step{step+1}.pt")

torch.save({
    "model": model.state_dict(),
    "step": STEPS,
    "L": L,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
}, OUT_DIR / "weight_net_final.pt")

print(f"\nDone. Saved to {OUT_DIR / 'weight_net_final.pt'}")
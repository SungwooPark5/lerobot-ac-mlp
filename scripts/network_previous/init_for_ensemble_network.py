"""Plan C: baseline ACT 체크포인트를 freeze 학습용으로 변환."""
import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

from pathlib import Path
from safetensors.torch import load_file

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.policies import PreTrainedConfig


BASELINE_DIR = Path(
    "/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model"
)
OUT_DIR = Path(
    "/home1/eunji24/lerobot_project/outputs/train/ensemble_network_init_100k_L50/pretrained_model"
)

# 1. config 로드 + Plan C용 수정
config = PreTrainedConfig.from_pretrained(str(BASELINE_DIR))
assert isinstance(config, ACTConfig), f"Expected ACTConfig, got {type(config)}"

config.use_learned_ensemble = True
config.ensemble_size = 50
config.n_action_steps = 1
config.freeze_action = True

# 2. 새 policy init
print("Initializing new policy with freeze_action=True...")
new_policy = ACTPolicy(config)

# 3. baseline weights 로드
print(f"Loading baseline weights from {BASELINE_DIR}")
baseline_sd = load_file(str(BASELINE_DIR / "model.safetensors"))

# 4. matching keys만 baseline에서 복사
new_sd = new_policy.state_dict()
loaded, missing = [], []

for k in new_sd:
    if k in baseline_sd and baseline_sd[k].shape == new_sd[k].shape:
        new_sd[k] = baseline_sd[k]
        loaded.append(k)
    else:
        missing.append(k)

new_policy.load_state_dict(new_sd)

print(f"\n  Loaded {len(loaded)} keys")
print(f"  Fresh init for {len(missing)} new keys (samples):")
for k in missing[:10]:
    print(f"    {k}")
print(f"    ... total {len(missing)}")

# 5. 새 ckpt 저장
OUT_DIR.parent.mkdir(parents=True, exist_ok=True)
new_policy.save_pretrained(str(OUT_DIR.parent))

print(f"\nSaved init ckpt to: {OUT_DIR.parent}")

# 6. Sanity check
print("\nSanity check: reload...")
config2 = PreTrainedConfig.from_pretrained(str(OUT_DIR.parent))
print(f"  freeze_action: {config2.freeze_action}")
print(f"  use_learned_ensemble: {config2.use_learned_ensemble}")
print(f"  ensemble_size: {config2.ensemble_size}")
print(f"  n_action_steps: {config2.n_action_steps}")
print("Done.")
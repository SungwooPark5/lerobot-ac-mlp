from pathlib import Path
import re

ROOT = Path(".").resolve()

ACM_60K_DIR = "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_transfer_cube_test_100k/checkpoints/060000/pretrained_model"
CACHE_PATH = "/home1/eunji24/lerobot_project/cache/acm_60k_chunks_vis/chunks.npz"

BASE_TRAIN = Path("train_weight_residual_acm_vis.py")
BASE_EVAL = Path("eval_with_weight_net_acm_vis.py")

if not BASE_TRAIN.exists():
    raise FileNotFoundError(BASE_TRAIN)
if not BASE_EVAL.exists():
    raise FileNotFoundError(BASE_EVAL)

train_template = BASE_TRAIN.read_text()
eval_template = BASE_EVAL.read_text()


def tag_from_coeff(coeff: float):
    return "pos" if coeff > 0 else "neg"


def fmt_lam(lam: float):
    return str(lam).replace(".", "p")


def patch_train(template: str, lam: float, coeff: float) -> str:
    tag = tag_from_coeff(coeff)
    lam_tag = fmt_lam(lam)
    out_dir = f"/home1/eunji24/lerobot_project/outputs/weight_network_acm_60k_residual_vis_{tag}_coeff_lam{lam_tag}"

    s = template

    s = re.sub(
        r'CACHE_PATH\s*=\s*Path\(".*?"\)',
        f'CACHE_PATH = Path("{CACHE_PATH}")',
        s,
    )
    s = re.sub(
        r'OUT_DIR\s*=\s*Path\(".*?"\)',
        f'OUT_DIR = Path("{out_dir}")',
        s,
    )
    s = re.sub(
        r'TE_COEFF\s*=\s*[-+]?\d*\.?\d+',
        f'TE_COEFF = {coeff}',
        s,
    )
    s = re.sub(
        r'LAMBDA_DELTA_REG\s*=\s*[-+]?\d*\.?\d+',
        f'LAMBDA_DELTA_REG = {lam}',
        s,
    )

    s = re.sub(
        r'name=f".*?"',
        f'name=f"acm_60k_vis_{tag}_coeff_lam{lam_tag}_L{{L}}_lambda{{LAMBDA_DELTA_REG}}"',
        s,
        count=1,
    )

    # checkpoint metadata에 base/coeff_tag 추가
    s = s.replace(
        '"base": "acm",',
        f'"base": "acm_60k",\n            "coeff_tag": "{tag}",',
    )
    s = s.replace(
        '"base": "acm",',
        f'"base": "acm_60k",\n    "coeff_tag": "{tag}",',
    )

    return s


def patch_eval(template: str, lam: float, coeff: float) -> str:
    tag = tag_from_coeff(coeff)
    lam_tag = fmt_lam(lam)
    out_dir = f"/home1/eunji24/lerobot_project/outputs/weight_network_acm_60k_residual_vis_{tag}_coeff_lam{lam_tag}"
    weight_path = f"{out_dir}/weight_net_final.pt"

    s = template

    s = re.sub(
        r'BASELINE_DIR\s*=\s*Path\(".*?"\)',
        f'BASELINE_DIR = Path("{ACM_60K_DIR}")',
        s,
    )
    s = re.sub(
        r'WEIGHT_NET_PATH\s*=\s*Path\(".*?"\)',
        f'WEIGHT_NET_PATH = Path("{weight_path}")',
        s,
    )
    s = re.sub(
        r'TE_COEFF\s*=\s*[-+]?\d*\.?\d+',
        f'TE_COEFF = {coeff}',
        s,
    )

    s = s.replace(
        'print("    ACM baseline n=50:      56%")',
        'print("    ACM 60k baseline n=50:  56%")',
    )
    s = s.replace(
        'print(f"    ACM residual vis L={L}: {pc_success * 100:.1f}%")',
        f'print(f"    ACM 60k residual vis {tag} coeff lam{lam} L={{L}}: {{pc_success * 100:.1f}}%")',
    )

    return s


# ------------------------------------------------------------
# pure TE eval 생성
# ------------------------------------------------------------
def make_pure_te(coeff: float) -> str:
    tag = tag_from_coeff(coeff)

    return f'''"""Pure TE eval for ACM 60k image-policy chunks. No weight network."""
import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import numpy as np
from collections import deque
from pathlib import Path

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task

BASELINE_DIR = Path("{ACM_60K_DIR}")

N_EPISODES = 50
BATCH_SIZE = 50
L = 100
TE_COEFF = {coeff}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PureTEPolicy:
    def __init__(self, baseline_policy, L=100, te_coeff=0.01):
        self.baseline = baseline_policy
        self.L = L
        self.te_coeff = te_coeff
        self.config = baseline_policy.config
        self.past_chunks = deque(maxlen=L)

    def reset(self):
        self.past_chunks.clear()

    def eval(self):
        self.baseline.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch):
        chunk = self.baseline.predict_action_chunk(batch)
        self.past_chunks.append(chunk)

        n = len(self.past_chunks)

        contribs = torch.stack([
            self.past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        logits = self.te_coeff * torch.arange(n, device=contribs.device, dtype=contribs.dtype)
        weights = torch.softmax(logits, dim=0)

        action = (weights.view(1, n, 1) * contribs).sum(dim=1)
        return action


def main():
    print(f"Loading ACM 60k baseline from {{BASELINE_DIR}}")
    baseline = ACMPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    policy = PureTEPolicy(
        baseline,
        L=L,
        te_coeff=TE_COEFF,
    ).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
        preprocessor_overrides={{
            "device_processor": {{"device": str(DEVICE)}},
        }},
    )

    from lerobot.envs.configs import AlohaEnv
    env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
    env_obj = make_env(env_cfg, n_envs=BATCH_SIZE, use_async_envs=False)

    while isinstance(env_obj, dict):
        print("make_env returned dict keys:", env_obj.keys())
        env_obj = list(env_obj.values())[0]

    env = env_obj
    print("final env type:", type(env))

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    print(f"\\nRunning ACM 60k pure TE rollout: coeff={{TE_COEFF}}, episodes={{N_EPISODES}}, batch={{BATCH_SIZE}}")

    seeds = list(range(N_EPISODES))[:BATCH_SIZE]
    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(BATCH_SIZE, dtype=bool)
    successes = np.zeros(BATCH_SIZE, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    policy.reset()

    step = 0
    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(obs_dict)
        obs = add_envs_task(env, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = postprocessor(action)
        action = env_postprocessor({{"action": action}})["action"]
        action_np = action.detach().to("cpu").numpy()

        obs_dict, reward, terminated, truncated, info = env.step(action_np)
        done = done | terminated | truncated

        if "final_info" in info:
            final_info = info["final_info"]

            if isinstance(final_info, dict):
                if "is_success" in final_info:
                    successes |= np.array(final_info["is_success"], dtype=bool)
            else:
                for i, fi in enumerate(final_info):
                    if isinstance(fi, dict) and fi.get("is_success", False):
                        successes[i] = True

        step += 1
        if step % 50 == 0:
            print(f"  step {{step}}/{{max_steps}}, done {{done.sum()}}/{{BATCH_SIZE}}")

    pc_success = successes.mean()

    print("\\nResults:")
    print(f"  pc_success = {{pc_success:.3f}} ({{successes.sum()}}/{{BATCH_SIZE}})")
    print("  Compare:")
    print("    ACM 60k baseline n=50:  56%")
    print(f"    ACM 60k pure TE {tag} coeff={{TE_COEFF}}: {{pc_success * 100:.1f}}%")

    env.close()


if __name__ == "__main__":
    main()
'''


# ------------------------------------------------------------
# 파일 생성
# ------------------------------------------------------------

jobs = [
    # positive coeff 추가분
    (1.0, 0.01),
    (0.1, 0.01),

    # negative coeff 전체
    (5.0, -0.01),
    (1.0, -0.01),
    (0.1, -0.01),
]

for lam, coeff in jobs:
    tag = tag_from_coeff(coeff)
    lam_tag = fmt_lam(lam)

    train_name = f"train_weight_residual_acm_60k_vis_{tag}_coeff_lam{lam_tag}.py"
    eval_name = f"eval_with_weight_net_acm_60k_vis_{tag}_coeff_lam{lam_tag}.py"

    Path(train_name).write_text(patch_train(train_template, lam, coeff))
    Path(eval_name).write_text(patch_eval(eval_template, lam, coeff))

    print("[WRITE]", train_name)
    print("[WRITE]", eval_name)

Path("eval_pure_te_acm_60k_pos_coeff.py").write_text(make_pure_te(0.01))
Path("eval_pure_te_acm_60k_neg_coeff.py").write_text(make_pure_te(-0.01))

print("[WRITE] eval_pure_te_acm_60k_pos_coeff.py")
print("[WRITE] eval_pure_te_acm_60k_neg_coeff.py")

print("\\nDone.")
print("\\n검사 명령:")
print('grep -n "ACTPolicy\\|modeling_act\\|last/pretrained_model\\|baseline_chunks" train_weight_residual_acm_60k_vis_* eval_with_weight_net_acm_60k_vis_* eval_pure_te_acm_60k_* || true')
print("\\n실행 순서:")
print("python eval_pure_te_acm_60k_pos_coeff.py")
print("python train_weight_residual_acm_60k_vis_pos_coeff_lam1p0.py")
print("python eval_with_weight_net_acm_60k_vis_pos_coeff_lam1p0.py")
print("python train_weight_residual_acm_60k_vis_pos_coeff_lam0p1.py")
print("python eval_with_weight_net_acm_60k_vis_pos_coeff_lam0p1.py")
print("python eval_pure_te_acm_60k_neg_coeff.py")
print("python train_weight_residual_acm_60k_vis_neg_coeff_lam5p0.py")
print("python eval_with_weight_net_acm_60k_vis_neg_coeff_lam5p0.py")
print("python train_weight_residual_acm_60k_vis_neg_coeff_lam1p0.py")
print("python eval_with_weight_net_acm_60k_vis_neg_coeff_lam1p0.py")
print("python train_weight_residual_acm_60k_vis_neg_coeff_lam0p1.py")
print("python eval_with_weight_net_acm_60k_vis_neg_coeff_lam0p1.py")

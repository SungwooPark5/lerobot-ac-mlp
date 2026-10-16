from pathlib import Path
import re

p = Path("eval_with_weight_net_vis_noisyobs.py")
s = p.read_text()

# argparse import 추가
if "import argparse" not in s:
    s = s.replace("import sys\n", "import sys\nimport argparse\n")

# parse_args 함수 추가
if "def parse_args():" not in s:
    insert = r'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_net_path", type=str, default=str(WEIGHT_NET_PATH))
    parser.add_argument("--eval_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--eval_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--state_noise_std", type=float, default=0.0)
    parser.add_argument("--state_noise_ratio", type=float, default=0.0)
    parser.add_argument("--image_noise_std", type=float, default=0.0)
    parser.add_argument("--image_noise_prob", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    return parser.parse_args()


def add_observation_noise(obs, args, step=None):
    # ===== state noise =====
    state_key = "observation.state"
    if state_key in obs and torch.is_tensor(obs[state_key]):
        if args.state_noise_std > 0.0 or args.state_noise_ratio > 0.0:
            state = obs[state_key]
            noise = torch.zeros_like(state)

            if args.state_noise_std > 0.0:
                noise = noise + args.state_noise_std * torch.randn_like(state)

            if args.state_noise_ratio > 0.0:
                scale = state.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
                noise = noise + args.state_noise_ratio * scale * torch.randn_like(state)

            obs[state_key] = state + noise

            if step == 0:
                print("[STATE NOISE DEBUG]")
                print("state mean abs:", state.abs().mean().item())
                print("state std mean:", state.std(dim=0).mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print("noise/state abs ratio:", (noise.abs().mean() / state.abs().mean().clamp_min(1e-6)).item())

    # ===== image noise =====
    if args.image_noise_std > 0.0:
        for key in list(obs.keys()):
            if not key.startswith("observation.images"):
                continue
            if not torch.is_tensor(obs[key]):
                continue

            img = obs[key]

            if args.image_noise_prob < 1.0:
                B = img.shape[0]
                apply_mask = torch.rand(B, device=img.device) < args.image_noise_prob
                view_shape = [B] + [1] * (img.ndim - 1)
                apply_mask = apply_mask.view(*view_shape).float()
                noise = args.image_noise_std * torch.randn_like(img) * apply_mask
            else:
                noise = args.image_noise_std * torch.randn_like(img)

            noisy_img = img + noise

            img_min = img.min().item()
            img_max = img.max().item()

            # 0~1 image면 clamp, normalize image면 clamp 안 함
            if img_min >= -0.1 and img_max <= 1.1:
                noisy_img = noisy_img.clamp(0.0, 1.0)

            obs[key] = noisy_img

            if step == 0:
                denom = img.abs().mean().clamp_min(1e-6)
                print(f"[IMAGE NOISE DEBUG] {key}")
                print("image min/max/mean:", img.min().item(), img.max().item(), img.mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print("noise/image abs ratio:", (noise.abs().mean() / denom).item())

    return obs

'''
    s = s.replace("\n\ndef main():", insert + "\n\ndef main():")

# main 시작에 args 추가
if "args = parse_args()" not in s:
    s = s.replace("def main():\n", "def main():\n    args = parse_args()\n    torch.manual_seed(args.noise_seed)\n    np.random.seed(args.noise_seed)\n")

# weight path arg 적용
s = s.replace('print(f"Loading weight network from {WEIGHT_NET_PATH}")', 'print(f"Loading weight network from {args.weight_net_path}")')
s = s.replace('ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)', 'ckpt = torch.load(Path(args.weight_net_path), map_location=DEVICE)')

# batch/episode arg 적용
s = s.replace('make_env(env_cfg, n_envs=BATCH_SIZE, use_async_envs=False)', 'make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)')
s = s.replace('print(f"\\nRunning rollout: {N_EPISODES} episodes, {BATCH_SIZE} parallel envs...")', 'print(f"\\nRunning rollout: {args.eval_episodes} episodes, {args.eval_batch_size} parallel envs...")')
s = s.replace('seeds = list(range(N_EPISODES))[:BATCH_SIZE]', 'seeds = list(range(args.eval_episodes))[:args.eval_batch_size]')
s = s.replace('done = np.zeros(BATCH_SIZE, dtype=bool)', 'done = np.zeros(args.eval_batch_size, dtype=bool)')
s = s.replace('successes = np.zeros(BATCH_SIZE, dtype=bool)', 'successes = np.zeros(args.eval_batch_size, dtype=bool)')

# preprocessor 뒤에 observation noise 추가
target = '''        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)
'''
replacement = '''        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        # baseline ACT가 chunk를 예측하기 전 observation에 noise 추가
        obs = add_observation_noise(obs, args, step=step)

        with torch.inference_mode():
            action = policy.select_action(obs)
'''
if target in s:
    s = s.replace(target, replacement)
elif "add_observation_noise(obs, args" not in s:
    raise RuntimeError("Could not find rollout obs preprocessing block.")

# 결과에 noise 설정 출력
if "image_noise_std =" not in s:
    s = s.replace(
        'print(f"  pc_success = {pc_success:.3f} ({successes.sum()}/{BATCH_SIZE})")',
        'print(f"  pc_success = {pc_success:.3f} ({successes.sum()}/{args.eval_batch_size})")\n'
        '    print(f"  state_noise_std = {args.state_noise_std}")\n'
        '    print(f"  state_noise_ratio = {args.state_noise_ratio}")\n'
        '    print(f"  image_noise_std = {args.image_noise_std}")\n'
        '    print(f"  image_noise_prob = {args.image_noise_prob}")'
    )
else:
    s = s.replace(
        'print(f"  pc_success = {pc_success:.3f} ({successes.sum()}/{BATCH_SIZE})")',
        'print(f"  pc_success = {pc_success:.3f} ({successes.sum()}/{args.eval_batch_size})")'
    )

p.write_text(s)
print("[OK] wrote eval_with_weight_net_vis_noisyobs.py")

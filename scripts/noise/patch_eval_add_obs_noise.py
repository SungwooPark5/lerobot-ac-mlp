from pathlib import Path
import re

p = Path("eval_with_weight_net_act_clean_noisyobs.py")
s = p.read_text()

# ------------------------------------------------------------
# 1) argparse 옵션 추가
# ------------------------------------------------------------
if "--image_noise_std" not in s:
    # state_noise 관련 parser.add_argument 뒤에 추가 시도
    anchor_patterns = [
        r'(parser\.add_argument\("--state_noise_ratio".*?\)\n)',
        r'(parser\.add_argument\("--state_noise_std".*?\)\n)',
    ]

    inserted = False
    for pat in anchor_patterns:
        m = re.search(pat, s)
        if m:
            insert = m.group(1) + (
                'parser.add_argument("--image_noise_std", type=float, default=0.0)\n'
                'parser.add_argument("--image_noise_prob", type=float, default=1.0)\n'
            )
            s = s[:m.start(1)] + insert + s[m.end(1):]
            inserted = True
            break

    if not inserted:
        raise RuntimeError("Could not find state_noise parser argument block.")

# ------------------------------------------------------------
# 2) add_observation_noise 함수 추가
# ------------------------------------------------------------
if "def add_observation_noise(" not in s:
    func = r'''

def add_observation_noise(obs, args, step=None):
    """
    baseline ACT가 chunk를 예측하기 전 observation에 noise를 넣는다.

    - observation.state:
        args.state_noise_std / args.state_noise_ratio 사용
    - observation.images.*:
        args.image_noise_std / args.image_noise_prob 사용

    중요:
    이 함수는 policy.select_action(obs) 전에 호출되어야 한다.
    그래야 baseline ACT의 predict_action_chunk가 noisy observation을 보고 chunk를 만든다.
    """

    # ========================================================
    # 1) state noise
    # ========================================================
    state_key = "observation.state"

    if state_key in obs and torch.is_tensor(obs[state_key]):
        if getattr(args, "state_noise_std", 0.0) > 0.0 or getattr(args, "state_noise_ratio", 0.0) > 0.0:
            state = obs[state_key]
            noise = torch.zeros_like(state)

            if getattr(args, "state_noise_std", 0.0) > 0.0:
                noise = noise + args.state_noise_std * torch.randn_like(state)

            # 주의: batch state std가 0이면 ratio noise가 거의 무효일 수 있음
            if getattr(args, "state_noise_ratio", 0.0) > 0.0:
                scale = state.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
                noise = noise + args.state_noise_ratio * scale * torch.randn_like(state)

            obs[state_key] = state + noise

            if step == 0:
                print("[STATE NOISE DEBUG]")
                print("state mean abs:", state.abs().mean().item())
                print("state std mean:", state.std(dim=0).mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print("noise/state abs ratio:", (noise.abs().mean() / state.abs().mean().clamp_min(1e-6)).item())

    # ========================================================
    # 2) image noise
    # ========================================================
    if getattr(args, "image_noise_std", 0.0) > 0.0:
        for key in list(obs.keys()):
            if not key.startswith("observation.images"):
                continue
            if not torch.is_tensor(obs[key]):
                continue

            img = obs[key]

            if getattr(args, "image_noise_prob", 1.0) < 1.0:
                B = img.shape[0]
                apply_mask = torch.rand(B, device=img.device) < args.image_noise_prob
                view_shape = [B] + [1] * (img.ndim - 1)
                apply_mask = apply_mask.view(*view_shape).float()
                noise = args.image_noise_std * torch.randn_like(img) * apply_mask
            else:
                noise = args.image_noise_std * torch.randn_like(img)

            noisy_img = img + noise

            # image가 0~1 범위면 clamp
            # normalize된 이미지면 clamp하지 않음
            img_min = img.min().item()
            img_max = img.max().item()

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

    # add_state_noise 함수가 있으면 그 뒤에 붙이기
    m = re.search(r'def add_state_noise\(.*?\n(?=\ndef |\nclass |\nif __name__|\ndef main|\n@|$)', s, flags=re.DOTALL)
    if m:
        s = s[:m.end()] + func + s[m.end():]
    else:
        # 없으면 main 앞에 붙이기
        m = re.search(r'\ndef main\(', s)
        if not m:
            raise RuntimeError("Could not find place to insert add_observation_noise().")
        s = s[:m.start()] + func + s[m.start():]

# ------------------------------------------------------------
# 3) rollout loop에서 add_state_noise(obs)를 add_observation_noise(obs,args,step)로 교체
# ------------------------------------------------------------
replace_count = 0

patterns = [
    r'obs\s*=\s*add_state_noise\(obs\)',
    r'obs\s*=\s*add_state_noise\(obs,\s*args\)',
]

for pat in patterns:
    new_s, n = re.subn(pat, 'obs = add_observation_noise(obs, args, step=step)', s)
    if n > 0:
        s = new_s
        replace_count += n

# add_state_noise 호출이 아예 없으면 preprocessor(obs) 바로 뒤에 삽입 시도
if replace_count == 0:
    pat = r'(obs\s*=\s*preprocessor\(obs\)\n)'
    m = re.search(pat, s)
    if not m:
        raise RuntimeError("Could not find obs = preprocessor(obs) or add_state_noise(obs) call.")
    insert = m.group(1) + '\n        obs = add_observation_noise(obs, args, step=step)\n'
    s = s[:m.start(1)] + insert + s[m.end(1):]

# ------------------------------------------------------------
# 4) 결과 print에 image noise 정보 추가
# ------------------------------------------------------------
if "image_noise_std =" not in s:
    # Results print 근처에 추가
    s = s.replace(
        'print(f"state_noise_ratio = {args.state_noise_ratio}")',
        'print(f"state_noise_ratio = {args.state_noise_ratio}")\n'
        '    print(f"image_noise_std = {args.image_noise_std}")\n'
        '    print(f"image_noise_prob = {args.image_noise_prob}")'
    )

p.write_text(s)
print("[OK] patched eval_with_weight_net_act_clean_noisyobs.py")
print("Added: --image_noise_std, --image_noise_prob")
print("Replaced add_state_noise with add_observation_noise before policy.select_action.")

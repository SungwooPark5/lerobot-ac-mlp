from pathlib import Path
import re

# ============================================================
# 1) image-feature train + direction files
# ============================================================
base_train = Path("train_weight_residual.py")
assert base_train.exists(), f"Missing {base_train}"

base_s = base_train.read_text()

lams = ["0.1", "1.0", "5.0"]

for lam in lams:
    out_file = Path(f"train_weight_residual_vis_pos_lam{lam}.py")
    s = base_s

    # output dir
    s = re.sub(
        r'OUT_DIR\s*=\s*Path\(".*?weight_network_residual_vis_lam5"\)',
        f'OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/weight_network_residual_vis_pos_lam{lam}")',
        s,
    )

    # lambda
    s = re.sub(
        r"LAMBDA_DELTA_REG\s*=\s*[0-9.]+",
        f"LAMBDA_DELTA_REG = {lam}",
        s,
    )

    # wandb name
    s = s.replace(
        'name=f"weight_residual_vis_lam5_L{L}_lambda{LAMBDA_DELTA_REG}"',
        f'name=f"weight_residual_vis_pos_lam{lam}_L{{L}}_lambda{{LAMBDA_DELTA_REG}}"',
    )

    # 핵심: TE prior 방향을 +로 변경
    s = s.replace(
        "-te_coeff * torch.arange(L, dtype=torch.float32)",
        "+te_coeff * torch.arange(L, dtype=torch.float32)",
    )

    # checkpoint metadata에 방향 명시 추가
    s = s.replace(
        '"lambda_delta_reg": LAMBDA_DELTA_REG,',
        '"lambda_delta_reg": LAMBDA_DELTA_REG,\n            "te_direction": "old_favored_pos",',
    )
    s = s.replace(
        '"lambda_delta_reg": LAMBDA_DELTA_REG,\n}',
        '"lambda_delta_reg": LAMBDA_DELTA_REG,\n    "te_direction": "old_favored_pos",\n}',
    )

    out_file.write_text(s)
    print("[OK] wrote", out_file)


# ============================================================
# 2) image-feature eval + direction file
# ============================================================
base_eval = Path("eval_with_weight_net_vis_noisyobs.py")
assert base_eval.exists(), f"Missing {base_eval}"

s = base_eval.read_text()

# weight net class 안의 기본 prior도 +로 변경
s = s.replace(
    "-te_coeff * torch.arange(L, dtype=torch.float32)",
    "+te_coeff * torch.arange(L, dtype=torch.float32)",
)

# warm-up pure TE 방향도 +로 변경
s = s.replace(
    "-self.te_coeff * torch.arange(n, device=contribs.device)",
    "+self.te_coeff * torch.arange(n, device=contribs.device)",
)

# 출력 문구 구분
s = s.replace(
    'print("\\nResults:")',
    'print("\\nResults: image-feature residual POS direction eval")',
)

Path("eval_with_weight_net_vis_noisyobs_pos.py").write_text(s)
print("[OK] wrote eval_with_weight_net_vis_noisyobs_pos.py")

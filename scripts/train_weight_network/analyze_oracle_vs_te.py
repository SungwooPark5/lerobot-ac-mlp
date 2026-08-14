# analyze_oracle_vs_te.py
import numpy as np
from pathlib import Path

CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")

L = 100
TE_COEFF = 0.01   # 네가 쓰는 pure TE 방향에 맞게. fresh-favored면 -0.01로 바꿔도 됨.

print(f"Loading {CACHE_PATH}")
data = np.load(CACHE_PATH)

chunks = data["chunks"]       # (N, L, action_dim)
actions = data["actions"]     # (N, action_dim)
episode_idx = data["episode_idx"]
ep_starts = data["ep_starts"] if "ep_starts" in data else None

N, H, A = chunks.shape
print("chunks:", chunks.shape)
print("actions:", actions.shape)

# TE weight
ages = np.arange(L)  # 0 old, L-1 newest라고 가정
# 후보 window는 [t-L+1 ... t] 순서일 가능성이 큼
# TE_COEFF > 0이면 older 쪽 가중, <0이면 newer 쪽 가중으로 해석하려면 아래처럼 reverse age 사용
logits = TE_COEFF * (L - 1 - ages)
w = np.exp(logits - logits.max())
w = w / w.sum()

te_errors = []
best_errors = []
newest_errors = []
mean_errors = []

best_indices = []
valid_count = 0

for t in range(N):
    ep = episode_idx[t]

    # 현재 timestep t에 기여 가능한 과거 chunk 후보들 모으기
    contribs = []

    for k in range(L):
        src = t - k
        if src < 0:
            continue
        if episode_idx[src] != ep:
            continue

        # src 시점에서 예측한 chunk 중 현재 t에 해당하는 action
        offset = k
        if offset >= H:
            continue

        contribs.append(chunks[src, offset])

    if len(contribs) == 0:
        continue

    contribs = np.stack(contribs, axis=0)  # old? 실제론 newest부터 쌓임
    target = actions[t]

    # newest = 가장 최근 src=t 후보, offset=0
    newest = chunks[t, 0]

    # 후보별 L1
    cand_err = np.abs(contribs - target).mean(axis=-1)
    best_err = cand_err.min()
    best_idx = cand_err.argmin()

    # 길이가 L보다 짧은 warmup은 해당 길이만큼 TE weight 재정규화
    m = len(contribs)

    # contribs는 k=0,1,2... 즉 newest -> older 순서
    # TE용으로 older -> newest 순서로 뒤집어서 weight 적용
    contribs_old_to_new = contribs[::-1]

    ww = w[-m:]
    ww = ww / ww.sum()

    te = (contribs_old_to_new * ww[:, None]).sum(axis=0)
    mean = contribs.mean(axis=0)

    te_err = np.abs(te - target).mean()
    newest_err = np.abs(newest - target).mean()
    mean_err = np.abs(mean - target).mean()

    te_errors.append(te_err)
    best_errors.append(best_err)
    newest_errors.append(newest_err)
    mean_errors.append(mean_err)
    best_indices.append(best_idx)
    valid_count += 1

te_errors = np.array(te_errors)
best_errors = np.array(best_errors)
newest_errors = np.array(newest_errors)
mean_errors = np.array(mean_errors)
best_indices = np.array(best_indices)

print("\n========== Oracle candidate analysis ==========")
print("valid frames:", valid_count)

def stat(name, x):
    print(f"{name:>12}: mean={x.mean():.6f}, median={np.median(x):.6f}, std={x.std():.6f}")

stat("TE", te_errors)
stat("Best cand", best_errors)
stat("Newest", newest_errors)
stat("Mean", mean_errors)

print("\n========== Pairwise win rate ==========")
print(f"Best < TE       : {(best_errors < te_errors).mean()*100:.2f}%")
print(f"Best +5% < TE   : {(best_errors < te_errors * 0.95).mean()*100:.2f}%")
print(f"Best +10% < TE  : {(best_errors < te_errors * 0.90).mean()*100:.2f}%")
print(f"TE < Newest     : {(te_errors < newest_errors).mean()*100:.2f}%")
print(f"Newest < TE     : {(newest_errors < te_errors).mean()*100:.2f}%")
print(f"Best < Newest   : {(best_errors < newest_errors).mean()*100:.2f}%")

print("\n========== Oracle improvement ==========")
improve = te_errors - best_errors
rel_improve = improve / (te_errors + 1e-8)

print(f"mean TE - Best error      : {improve.mean():.6f}")
print(f"median TE - Best error    : {np.median(improve):.6f}")
print(f"mean relative improvement : {rel_improve.mean()*100:.2f}%")
print(f"median relative improvement: {np.median(rel_improve)*100:.2f}%")

print("\n========== Best candidate position ==========")
# best_idx는 newest 기준: 0이면 현재 chunk, 클수록 더 오래된 chunk
print(f"best idx mean new=0      : {best_indices.mean():.2f}")
print(f"best idx median new=0    : {np.median(best_indices):.2f}")
for i in [0, 1, 2, 5, 10, 20, 50, 99]:
    print(f"best_idx == {i:2d}: {(best_indices == i).mean()*100:.2f}%")

print("\nInterpretation:")
if (best_errors < te_errors * 0.90).mean() > 0.30:
    print("=> oracle 후보 선택 이득이 큼. weight-net/selector 필요성 있음.")
elif (best_errors < te_errors * 0.95).mean() > 0.30:
    print("=> 후보 선택 이득은 있음. 다만 네트워크가 그걸 학습 가능한지 별도 확인 필요.")
else:
    print("=> oracle best도 TE보다 크게 좋지 않음. 이 세팅은 selection 문제가 아닐 가능성이 큼.")
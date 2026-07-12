# notebooks/ — 논문 실험 파이프라인 (Action Chunking with Mamba)

이 폴더 하나로 **학습 → eval → 표/그림**이 전부 재현된다. 브랜치 = `ecd-final`.

## 프로토콜 (전 그룹 공통)

| | |
|---|---|
| **학습** | **150k step** · **seed 4개**(0-3) · **lr sweep 없음** (전 모델 고정 **1e-5**) |
| **eval** | **150k 체크포인트 1개**를 **5회 반복** — rep 마다 `--seed = 1000 + 100·rep` → env 초기상태가 달라짐 |
| **수치** | 모델당 4 seed × 5 rep = **20 run** → **mean ± std** + pooled Wilson 95% CI |
| **왜 best-ckpt 를 안 고르나** | 모델마다 유리한 step 을 뽑으면 cherry-pick → 전 모델 동일 step(150k) 고정 |
| **왜 rep 마다 seed 를 바꾸나** | 같은 seed 로 5번 돌리면 결정적이라 반복이 무의미. 학습 seed(모델 분산)와 rep(평가 분산)을 분리 |

예외: `diffusion` / `smolvla` 만 **각자 원 논문 lr(1e-4)** — 남의 방법을 우리 lr 로 깎으면 baseline 이 불공정.

## 노트북 (모델 그룹마다 학습/eval 이 따로)

| | 학습 | eval | 모델 | 잡 / run |
|---|---|---|---|---|
| preflight | `00_smoke` | — | — | CUDA·mamba_ssm·**carry parity 테스트**·dry-run |
| **★우리 모델** | `01_train_ours` | `02_eval_ours` | **`ours`** | 4잡 / 20 run |
| **★대조군** | `03_train_acm` | `04_eval_acm` | `acm` (carry off) | 4잡 / 20 run |
| baseline | `05_train_baseline` | `06_eval_baseline` | `act`·`diffusion`·`smolvla`·`acm2` (+`act_te` eval만) | 16잡 / 100 run |
| ablation | `07_train_ablation` | `08_eval_ablation` | `acm_carry`·`acm_bimamba`·`acm_s7` (+표) | 12잡 / 60 run |
| 리포트 | — | `09_report_sr` | 전 모델 SR 표·그림 (Table 1) | — |
| | — | `10_report_jerk` | 떨림 — 경계/내부 jerk, SPARC, **경계정렬 jerk 프로파일**(헤드라인 그림) | — |
| | — | `11_efficiency` | latency·VRAM vs K (어텐션 O(L²) vs Mamba O(L)). **학습 불필요** | — |
| 보조 | `20_train_libero_seed{0-3}` | `21_eval_libero` · `21b_*` | LIBERO-10 | — |

**`01`(ours) 다음은 `03`(acm).** 우리 헤드라인 주장이 "plain Mamba 디코더 대비 [XX]p 개선"이라,
같은 백본에서 carry 를 끈 `acm` 이 **그 수치의 분모**다. 없으면 개선폭을 못 쓴다.
둘 다 4잡이라 8 GPU 면 `01`+`03` 을 동시에 돌려도 된다.

## 모델

| 그룹 | 태그 |
|---|---|
| baseline | `act` · `act_te`(= act ckpt + eval-time TE, 학습 X) · `diffusion` · `smolvla` · `acm2`(Mamba-2 dec) · `acm`(Mamba-1 dec, carry off) |
| **ours** | `ours` = `acm` + **carry(SSCP)** + **BiMamba** + **MOSAIC**(overlap-add crossfade) |
| ablation 사다리 | `acm` → `acm_carry` → `acm_bimamba` → `acm_s7` → `ours` |

## 실행

```bash
# 0) carry 정확성 먼저 (실패하면 ours 학습이 무의미)
python tests/test_acm_sscp_literal.py

# 1) 노트북
cd notebooks && jupyter lab      # 00 -> 01/02(ours) -> 03/04(acm) -> 05/06, 07/08 -> 09/10/11
```

- **끊겨도 안전**: 학습은 `--resume` 자동, eval 은 끝난 run(`eval_info.json`) 자동 skip.
- **seed 분담**: 각 노트북 상단 `SEEDS` 만 바꾸면 됨 (예: 해준 `[0,1]` / 은지 `[2,3]`).
  결과가 같은 경로에 쌓여 리포트에서 **자동 pooled**.

## 설정 (`common_final.py` 상단)

```python
cf.STEPS        # 150_000    학습 스텝
cf.CKPT_STEP    # 150_000    평가할 체크포인트
cf.MAIN_SEEDS   # [0,1,2,3]  학습 seed
cf.EVAL_REPEATS # 5          반복 eval 횟수
cf.EVAL_N_EP    # 50         rep 1회당 에피소드 (부담되면 30)

cf.GROUP_OURS     # ['ours']
cf.GROUP_ACM      # ['acm']                                   ← 대조군
cf.GROUP_BASELINE # ['act', 'diffusion', 'smolvla', 'acm2']
cf.GROUP_ABLATION # ['acm_carry', 'acm_bimamba', 'acm_s7']
```

## 경로

| | 기본값 | 오버라이드 |
|---|---|---|
| 레포 | 이 파일 위치에서 자동 유도 | `LEROBOT_REPO` |
| 학습/eval subprocess python | `~/lerobot_project/lerobot_env/bin/python` | `LEROBOT_PYTHON` |
| 출력 | `~/lerobot_project/outputs/final/` | `LEROBOT_OUTPUT` |

출력(체크포인트·영상·action .pt)은 **레포 밖**에 쌓인다 → git 오염 없음.

## 의존성

- `mamba-ssm` (CUDA 전용 커널) — Mamba 계열 정책 전부 필요
- LIBERO eval 만 별도 시뮬 설치 필요 (보조 실험. 설치 노트북은 레포에 포함하지 않음)
- 데이터셋은 HF 참조: `lerobot/aloha_sim_insertion_human` · `HuggingFaceVLA/libero` · `eejjii/*`(so-101)

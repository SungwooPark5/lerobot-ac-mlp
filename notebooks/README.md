# notebooks/ — 논문 실험 파이프라인 (Action Chunking with Mamba)

이 폴더 하나로 **학습 → eval → 표/그림**이 전부 재현된다. 브랜치 = `ecd-final`.

## 폴더 구성

```
notebooks/
  common_final.py · common_v23.py · smooth_metrics.py   드라이버(수정 대상 아님)
  00_smoke.ipynb                    preflight — 여기부터
  insertion/   메인(긴 horizon)    01~08  학습/eval (ours·acm·baseline·ablation)
  transfer/    짧은 앵커            12~17 (GPU 2장씩 a/b) · 19a/19b(4-GPU 한번에) · 0e/0r/0s
  libero/      보조(LIBERO-10)      eval_node1_4gpu · eval_node2_4gpu  ← 학습된 4모델 eval (GPU 4+4)
  reports/     표·그림              09_sr · 10_jerk · 11_efficiency · 18_horizon
  utils/       유틸                 0l_collect_libero · 0m_rename_s7_to_mosaic · 0v_record_videos
```

> 어느 하위폴더에서 열어도 `import common_final` 이 자동으로 상위 `notebooks/` 를 찾는다 — cwd 신경 쓸 필요 없음.

## 프로토콜 (전 그룹 공통)

| | |
|---|---|
| **학습** | **150k step** · **seed 4개**(0-3) · **lr sweep 없음**(고정 1e-5) · **학습중 eval OFF** |
| **eval** | **150k 체크포인트 1개**를 **5회 반복 × 500 에피소드** — rep 마다 `--seed = 1000 + 100·rep` → env 초기상태가 달라짐 |
| **수치** | 모델당 4 seed × 5 rep = **20 run × 500ep = 10,000 에피소드** → **mean ± std** + pooled Wilson 95% CI |
| **왜 best-ckpt 를 안 고르나** | 모델마다 유리한 step 을 뽑으면 cherry-pick → 전 모델 동일 step(150k) 고정 |
| **왜 rep 마다 seed 를 바꾸나** | 같은 seed 로 5번 돌리면 결정적이라 반복이 무의미. 학습 seed(모델 분산)와 rep(평가 분산)을 분리 |
| **왜 학습중 eval 을 끄나** | 어차피 150k 한 점만 쓰므로 불필요. 학습 프로세스가 sim env 를 아예 안 만들어 **시간·VRAM 절약**(옛 CUDA OOM 원인 제거) |

예외: `diffusion` / `smolvla` 만 **각자 원 논문 lr(1e-4)** — 남의 방법을 우리 lr 로 깎으면 baseline 이 불공정.

## `insertion/` — 메인 (긴 horizon, GPU 4장)

| | 학습 | eval | 모델 | 잡 / run |
|---|---|---|---|---|
| preflight | `00_smoke` (루트) | — | — | CUDA·mamba_ssm·**carry parity 테스트** |
| **★우리 모델** | `01_train_ours` | `02_eval_ours` | **`ours`** | 4잡 / 20 run |
| **★대조군** | `03_train_acm` | `04_eval_acm` | `acm` (carry off) | 4잡 / 20 run |
| baseline | `05_train_baseline` | `06_eval_baseline` | `act`·`diffusion`·`smolvla`·`acm2` (+`act_te` eval만) | 16잡 / 100 run |
| ablation | `07_train_ablation` | `08_eval_ablation` | `acm_carry`·`acm_bimamba`·`acm_mosaic` (+표) | 12잡 / 60 run |

## `transfer/` — 짧은 앵커 (GPU 2장씩 두 창 `a`/`b` 로 분할)

`a` = seed 0,1 · `b` = seed 2,3. 두 창은 다른 GPU 를 쓰므로 동시에 띄워도 안 밟는다(`cf.part('A'/'B')`).

| | 학습 | eval | 모델 | 창당 잡 / run |
|---|---|---|---|---|
| 우리 모델 | `12a` · `12b` | `13a` · `13b` | `ours` | 2잡 / 10 run |
| 대조군 | `14a` · `14b` | `15a` · `15b` | `acm` | 2잡 / 10 run |
| baseline | `16a` · `16b` | `17a` · `17b` | act·diffusion·smolvla·acm2 (+act_te) | 8잡 / 50 run |
| **한번에(4-GPU)** | `19a_train_all` | `19b_eval_all` | `TAGS`/`TASK` 로 선택, 4 seed 동시 | — |
| 이어학습 | `0r_resume_train` | — | 끊긴 학습 → 마지막 ckpt 부터 (끝난 seed skip) | — |
| 2-GPU eval | — | `0e_eval_transfer_2gpu` | seed 4개 × 5 rep = 20 run (GPU 2장) | 20 run |
| 팀 공유 | — | `0s_share_transfer` | 결과 → 표·그림·요약 → **zip 하나** | — |

## `libero/` — 보조 eval (LIBERO-10, **이미 학습된 4모델**, 노드 2대 GPU 4+4)

학습은 안 한다 — **기존 체크포인트로 eval 만**. 학습된 4모델 = `acm` · `act` · `bimamba` · `bimamba_s7`.

| 노트북 | GPU | eval 몫 |
|---|---|---|
| `eval_node1_4gpu` | 4 | seed 0·1 (8 eval) |
| `eval_node2_4gpu` | 4 | seed 2·3 (8 eval) |

- 전체 4모델 × 4 seed(0-3) = **16 eval** 을 GPU 비(4:4)로 8/8 나눔(`cf.split_by_gpu`).
- eval = **150k ckpt × 500ep** + action(.pt)→떨림(jerk/LDJ/SPARC/SignFlip). **LIBERO 시뮬 필요**.
- 체크포인트 없거나 이미 끝난 eval 은 자동 skip → 재실행 안전. 결과는 `eval_clean/libero_10/…` 에 모여 리포트 자동 pooled.
- `bimamba_s7` 은 rename 전 폴더명 그대로 사용(노트북이 태그를 자동 등록). 나중에 `utils/0m` 돌리면 mosaic 으로 통일.

## `utils/`

| 노트북 | 하는 일 |
|---|---|
| `0l_collect_libero` | `eval_clean/libero_10` 스캔 → SR 표·run 목록·떨림(jerk/LDJ/SPARC/SignFlip) → **zip** |
| `0m_rename_s7_to_mosaic` | 서버 결과 폴더 중 이름에 `s7` 든 것 → `mosaic` (dry-run→충돌점검→실행→검증) |
| `0v_record_videos` | 150k ckpt 로 에피소드 mp4 저장 (기존 SR/떨림 결과와 분리) |

## `reports/` — 표·그림

| 노트북 | 내용 |
|---|---|
| `09_report_sr` | SR 표·그림 (Table 1). `TASK` 만 바꾸면 transfer/LIBERO 도 |
| `10_report_jerk` | 떨림 — 경계/내부 jerk, SPARC, **경계정렬 jerk 프로파일**(헤드라인 그림) |
| `11_efficiency` | latency·VRAM vs K (어텐션 O(L²) vs Mamba O(L)). **학습 불필요** |
| **`18_report_horizon`** | **SR vs task horizon** (transfer → insertion) — "길수록 격차가 벌어지는가" |

**task 축**: `transfer`(짧) → `insertion`(긺) → `libero_10`(보조).
Intro 가 *"짧은 task 는 ACT 와 대등, 길수록 우리가 앞선다"* 고 주장하므로 **짧은 쪽(transfer) 도 필요**하다.
ablation 은 메인(insertion)에서만 돌린다.

**`01`(ours) 다음은 `03`(acm).** 우리 헤드라인 주장이 "plain Mamba 디코더 대비 [XX]p 개선"이라,
같은 백본에서 carry 를 끈 `acm` 이 **그 수치의 분모**다. 없으면 개선폭을 못 쓴다.

## 모델

| 그룹 | 태그 |
|---|---|
| baseline | `act` · `act_te`(= act ckpt + eval-time TE, 학습 X) · `diffusion` · `smolvla` · `acm2`(Mamba-2 dec) · `acm`(Mamba-1 dec, carry off) |
| **ours** | `ours` = `acm` + **carry(SSCP)** + **BiMamba** + **MOSAIC**(overlap-add crossfade) |
| ablation 사다리 | `acm` → `acm_carry` → `acm_bimamba` → `acm_mosaic` → `ours` |

## 실행

```bash
# 0) carry 정확성 먼저 (실패하면 ours 학습이 무의미)
python tests/test_acm_sscp_literal.py

# 1) 노트북
cd notebooks && jupyter lab
#   00_smoke -> insertion/01,02(ours) -> 03,04(acm) -> 05~08 -> reports/09,10,11
#   LIBERO 는 세 노드에서 libero/train_node{A,B,C} (그다음 eval_node{A,B,C})
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
cf.EVAL_N_EP    # 500        rep 1회당 에피소드
cf.v23.EVAL_FREQ # 0         학습중 eval OFF (수렴 곡선 원하면 10_000)

cf.GROUP_OURS     # ['ours']
cf.GROUP_ACM      # ['acm']                                   ← 대조군
cf.GROUP_BASELINE # ['act', 'diffusion', 'smolvla', 'acm2']
cf.GROUP_ABLATION # ['acm_carry', 'acm_bimamba', 'acm_mosaic']
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
- LIBERO **eval** 만 별도 시뮬(robosuite/libero) 설치 필요 (학습은 데이터셋만. 설치 노트북은 레포에 포함하지 않음)
- 데이터셋은 HF 참조: `lerobot/aloha_sim_insertion_human` · `HuggingFaceVLA/libero` · `eejjii/*`(so-101)

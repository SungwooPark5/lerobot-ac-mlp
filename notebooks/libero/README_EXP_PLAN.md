# 실험 계획 — 2026-08-24 (UBAI 복구 후)

은지님 8/24 요청 + 8/18 교수님 지시에 대한 실행 계획. **서버에서 `git pull` 후 이 순서대로.**

---

## 배경 — 지금 뭘 증명해야 하나

교수님 지시(8/18) 세 갈래:
1. overlap 떼고 **ACT가 무너지는 지점**을 찾는다 → 거기서 BiMamba+carry가 의미 있으면 논문을 그쪽으로 다시 쓴다
2. 없으면 둘 중 하나만 살려서 의미 있는 경우를 찾는다
3. 그것도 없으면 현 상태로 마무리

은지님 8/18 결론:
- **Smoothness는 접는다. Temporal Ensemble을 이길 수 없다.**
- 긴 chunk에서 ACT가 무너지고, 거기선 BiMamba가 산다
- 짧은 stride에서는 carry가 산다
- **각각 다른 조건에서 되는데 같이 쓰면 안 된다**

---

## 오늘 밤 실행 (6 GPU = 3노드 x 2 GPU) — **여기부터**

`exp5_tonight.py` + `nA_tonight.ipynb` / `nB_tonight.ipynb` / `nC_tonight.ipynb`.
아래 P1~P3은 그 노트북들의 잡 우선순위로 **이미 들어가 있다.** 노트북이 인벤토리를 스캔해서
학습 안 된 것은 학습 큐로, 된 것은 eval 큐로 자동 분류한다.

| 노드 | GPU | 기본 역할 |
|---|---|---|
| A | 0,1 | 학습 2잡 (~15h, 내일 오후 해금) |
| B | 0,1 | eval (우선순위 짝수 index) |
| C | 0,1 | eval (우선순위 홀수 index) |

역할은 고정이 아니다. `X.suggest()` 가 ready eval 큐 깊이를 보고 조정한다
(≥24셀이면 B·C 둘 다 eval, 1~23셀이면 C도 학습, 0셀이면 셋 다 학습).
**출력이 "기본값과 다르다"고 하면 알려주는 `X.ROLE_OVERRIDE = ...` 한 줄을 세 노트북 모두에 넣어야
잡이 안 겹친다.**

예산: eval 1셀(LIBERO-10 x 50ep/task = 500ep) ≈ 2 GPU-h / 학습 1잡(150k) ≈ 15 GPU-h
(해준님 8/11 실측 "500ep 2시간, 5000ep 하루").

---

## 실행 순서

### P1 (최우선) — TE 축, 크로스오버 K 찍기

⚠️ **이 문서의 이전 버전은 "act+TE 는 K 5개를 학습해야 한다"고 썼는데 틀렸다.**
`common_final.act_te_eval_cmd` 가 하듯 **TE 는 ACT 체크포인트에 플래그만 얹으면 된다**
(`--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1`). **ACT ckpt 가 있는 K 는 재학습 0.**
ACT ckpt 는 K=50, K=100 이 이미 있다.

그리고 K=100 은 **이미 측정돼 있다** (8/18 슬라이드 88, overlap off):

| K | act+TE | bimamba+TE | bimamba+carry+TE |
|---|---|---|---|
| 10 | ? | 75.4 | 74.3 |
| 15 | ? | 75.6 | 76.4 |
| 20 | ? | 71.4 | 71.9 |
| 50 | **오늘 밤 (eval만)** | 64.1 | 63.6 |
| 100 | **30.5 ✅** | **49.3 ✅** | 50.2 |
| 150 | ACT K=150 학습 후 | 26.8 | 27.1 |

즉 **"ACT 가 무너지는 지점"은 이미 손에 있다** — K=100, TE on 에서 +18.8pt.
지금 할 일은 그걸 *발견*하는 게 아니라 **크로스오버 K 를 찍는 것**이다.
K=50 에서 act+TE 가 이기고 K=100 에서 뒤집히면 그게 논문 그림 1이다.

### P2 — `exp4_regime_map` 축 (K × 실행 stride, TE off)

변형 4종(ACT / carry / bimamba / bimos). stride 는 추론 파라미터라 **재학습 없이 eval 플래그만**
바꾸면 된다 → 기존 ckpt 로 커버. 8/18 가설("BiMamba 는 긴 stride, carry 는 짧은 stride")을
하나의 표로 확정하는 게 목적.

### P3 — 순수 BiMamba 학습 (`bimamba_pure`)

아래 "교란 2" 참조. `exp5_tonight` 의 학습 큐 **1순위**로 이미 올라가 있다
(K=100 부터, 이어서 K=50/150/20).

---

## 반드시 알고 있어야 할 제약·교란 3가지

### 제약 1 — TE와 stride는 같은 축에 못 놓는다

`src/lerobot/policies/acm2/configuration_acm2.py`:

```python
if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
    raise NotImplementedError(
        "`n_action_steps` must be 1 when using temporal ensembling. ...")
```

**TE ⟹ stride=1.** 그래서 "TE + stride 75" 같은 조합은 애초에 불가능하다.
- TE 축 = `exp3_*_te_ksweep` (stride 1 고정, K만 변화)
- stride 축 = `exp4_regime_map` (TE off)

두 노트북을 나눈 이유가 이것이다.

### 교란 2 — "BiMamba only"가 순수하지 않다

`common_v23.py`의 `MODEL_CONFIGS`에 **carry 없는 BiMamba 태그가 없다.**

```python
"bimamba": ("acm2_sscp_literal_bimamba", LR, 100, [_CARRY_ON], True),
#                                                  ^^^^^^^^^^  ^^^^
#                                                  carry on    use_chunk_pairs
```

8/18에 은지님이 "bimamba only를 전부 켜고 돌려서 다시 측정해야겠다"고 하신 그 문제가 코드에 그대로 있다.
eval에서 `sscp_enabled=false`를 걸어도 **학습은 chunk-pair로 된 상태**라 순수 BiMamba가 아니다.

→ `exp5_tonight`은 두 갈래로 간다.
  - **오염된 프록시**(싼 것): 기존 ckpt에 `sscp_enabled=false`만 걸어 eval. 라벨 = `bimamba_cpoff`.
    표에 그대로 남겨 오염 사실을 숨기지 않는다.
  - **순수판**(비싼 것): `bimamba_pure` 태그 = `use_chunk_pairs=False` + `_CARRY_OFF`로 **새로 학습**.
    학습 큐 1순위. 노드 A의 dry-run에서 `--use_chunk_pairs`가 **없는지** 반드시 눈으로 확인할 것.

### 교란 3 — 기존 표에 libero/aloha가 섞여 있다

8/18 랩미팅 슬라이드 93(libero-10, K=100)과 94(aloha, K=100)의 **stride 10 / 50 / 75 행이 완전히 동일**하다.

```
stride 10 : 2.0 / 22.5 / 15.5 / 17.5   <- 두 슬라이드 동일
stride 50 : 76.0 / ... / ... / 47.0    <- ACT·BiMOS 동일
stride 75 : 62.0 / 49.5 / 75.0 / 51.0  <- 두 슬라이드 동일
```

교차검증이 되는 건 **LIBERO K=100 stride=100 행 하나뿐**이다
(ACT 18.8 / ACM2 25.3 / BiMamba 31.9 / BiMOS 25.6 — 슬라이드 88의 s0 열과 일치).

따라서:
- "제일 높았던 bimamba only chunk100 stride75 = 75.0"은 **ALOHA 값**이다
- LIBERO 최고 67.6%는 **ACT**(K=50, stride 10)의 값이지 우리 모델이 아니다
- 결론 슬라이드의 "LIBERO K=100 s10 BiMamba 65.0 vs ACT 42.5"는 슬라이드 93(ACT 2.0 / BiMamba 15.5)과도 안 맞는다

**그래서 exp4에서 LIBERO를 처음부터 다시 뜬다.**

---

## 결과에 따른 분기

| 결과 | 논문 방향 |
|---|---|
| `bimamba+TE` > `act+TE` 인 K 구간 존재 | **"긴 chunk 레짐에서 ACT의 붕괴"**로 재작성. TE를 기본 세팅으로 깔고 K를 축으로. MOSAIC·smoothness는 부록 |
| TE 축은 안 되는데 stride 축에서 carry가 우세 | carry 쪽으로 축 이동 |
| 둘 다 되는데 합치면 안 됨 | **레짐 논문**. "두 메커니즘은 보완재가 아니라 대체재"가 기여 |
| 아무것도 ACT를 못 이김 | 교수님 3번 분기 |

---

## 논문 쪽에서 병행해야 할 것 (실험과 별개)

현재 `sections/`에 남아 있는 문제들:

1. **본 표에 `ACM2 + BiMamba` 행이 없다.** 어블레이션 표에는 있고 거기서 BiMamba 단독이
   LIBERO 37.5 / ALOHA 60.56으로 **전체 모델(33.9 / 59.72)보다 높다.** 본 표만 보면 BiMOS가
   최고인 것처럼 보이는데 리뷰어가 어블레이션을 펴면 바로 걸린다.
2. **TE 비교가 논문에 한 줄도 없다.** MOSAIC의 경쟁자가 TE인데 `compared methods`에 없다.
3. `experiments.tex`는 "5 seeds(0–4), 500 test episodes"라고 쓰여 있는데 실제 데이터는
   seed 1~2개, n=100/200/1000이 섞여 있다.
4. `[XX]` 플레이스홀더 5곳 (`conclusion.tex` 2, `experiments.tex` 1, `intro.tex` 2)
5. 제목이 갈려 있다 — `\shorttitle{BiMOS}` vs `\title{MambaBridge: ...}`, 본문·highlights는 BiMOS

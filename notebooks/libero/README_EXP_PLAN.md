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

## 8/25 재설계 — 승리 조건: 토너먼트 — **여기부터**

`exp5_tonight.py` + `nA/nB/nC_tonight.ipynb`. 노트북이 인벤토리를 스캔해서
학습 안 된 것은 학습 큐로, 된 것은 eval 큐로 자동 분류한다. 세 노트북은 **내용이 같고
노드 문자만 다르다** (6·7번 셀이 배정된 역할이 아니면 스스로 건너뛴다).

"ACT 를 넘는다" 는 두 가지이고 둘 다 명시적으로 노린다:

- **W1. 동일 세팅 우위** — 같은 (K, stride, TE) 에서 ours > ACT.
  이미 있다: K=100 TE +18.8 (49.3 vs 30.5), K=100 s100 TE-off +10.1.
- **W2. 토너먼트 우위** — 각 방법에 **같은 탐색 예산**(K × stride × TE on/off)을 주고
  방법별 best 끼리 비교했을 때 1등. 은지님 8/24 "67.6 을 넘을 값"이 이것.
  우리 best 만 재고 ACT 는 한 세팅만 재면 체리피킹이라 리뷰에서 깨진다.

### 8/25 오후 — 판정 2건

**① stride sweep 슬라이드는 aloha 오염 확정** (은지님 8/24: "76 까지 나온 건 libero 에는 없다,
67.6 이 최고"). libero/aloha 슬라이드의 s=10/50/75 행이 동일하다. 따라서:
- ACT 76.0 (K=100 s50) ❌, BiMamba 75.0 (K=100 s75) ❌ — libero 근거로 사용 금지
- 교차검증된 libero 행은 K=100 s=100 하나뿐 (ACT 18.8 / ACM2 25.3 / BiMamba 31.9 / BiMOS 25.6)
- **ACT 의 진짜 libero 최고 = 67.6 (K=50 s10, TE off)** — 이게 기준선
- **레짐 맵(rm_* 셀) = 이 실험의 UBAI 재구현.** 은지님이 요청한 "해당 실험 다시 구현"이 이것

**② act+TE 열 측정 완료** — `exp3_act_te_ksweep` (ak{K}_te 태그, 100ep/task, seed0, 동일 UBAI).
act_k10/15 는 이미 학습돼 있었다(학습 셀 전부 skip):

| K | act+TE | bimamba+TE (cpoff) | +carry (cpoff) | 마진 |
|---|---|---|---|---|
| 10 | **72.5** ← ACT 최고 | 75.4 | 74.3 | +2.9 |
| 15 | 67.3 | 75.6 | **76.4** ← 전체 최고 | +9.1 |
| 20 | 61.2 | 71.4 | 71.9 | +10.7 |
| 50 | 39.7 | 64.1 | 63.6 | +24.4 |
| 100 | 27.7 | 49.3 | 50.2 | +22.5 |
| 150 | 21.9 | 26.8 | 27.1 | +5.2 |

**전 K 에서 우리가 위. max(ours)=76.4 vs max(ACT)=72.5 → 100ep 기준 W2 승리.**
ACT 는 K 에 대해 단조 붕괴(72.5→21.9), 우리는 K=20 까지 71+ 유지 — 크로스오버 그림도 성립.

남은 확정 작업: (1) K=10/15/20 을 **500ep 재측정** (bimamba 쪽 n=100, SE ±4.3 이라 K=10/15
마진이 오차 안 — exp5 의 te_* 셀이 그 역할), (2) **순수판 `bimamba_pure_k10/15` 학습+측정**
(현재 bimamba 값은 carry-학습 오염판), (3) K=50 s10 레짐 행 (67.6 과 TE 없는 정면 승부).

역할은 `X.suggest()` 가 정한다 (8/25 규칙):
1. **critical 학습(act/pure @ K=10/15)이 노드를 먼저 선점** — 판정 셀을 여는 학습은
   eval 큐 소화보다 가치가 높다. 끝나면 선점이 자동 해제.
2. 남는 노드는 ready eval 큐를 포화.
3. 그래도 남으면 일반 학습.

**출력이 "기본값과 다르다"고 하면 알려주는 `X.ROLE_OVERRIDE = ...` 한 줄을 세 노트북 모두에 넣어야
잡이 안 겹친다.**

예산: eval 1셀(LIBERO-10 x 50ep/task = 500ep) ≈ 2 GPU-h / 학습 1잡(150k) ≈ **8 GPU-h**
(8/24 밤 실측: 4잡을 2 GPU 로 2배치 = 약 16h).

잔존 리스크: 500ep 재측정에서 K=10/15 마진(+2.9/+9.1)이 뒤집힐 수 있다(현재 오차 안).
그 경우 K=20/50 마진(+10.7/+24.4)으로 후퇴. 그것도 흔들리면 폴백 = W1(긴 chunk 크로스오버 +
stride 큰 레짐에서만 성립하는 추론-호출-절약 효율 주장, TE 는 매 스텝 추론이라 효율 주장 불가)
+ real-world(SO-101 sorting·OOD).

---

## 실행 순서

### P1 (최우선) — TE 토너먼트 축, **500ep 확정 런**

100ep 판정은 위 "8/25 오후" 표로 끝났다. 이제 exp5 의 te_* 셀이 같은 비교를 500ep 로
다시 잰다 (exp3 는 ak{K}_te 태그로 저장돼 있어 exp5 가 te_* 태그로 다시 돈다 — 중복이
아니라 n 업그레이드: SE ±4.3 → ±2.2). K 순서 = 마진이 얇아 확정이 급한 순:
**10**(+2.9) → **15**(+9.1, max 자리) → 20 → 50 → 100 → 150.
K=10/15 는 act / 순수 bimamba 2종만, 나머지 K 는 5종 전부.
순수판 te 셀(te_bimamba_*)은 `bimamba_pure_k10/15` 학습이 끝나는 대로 자동 편입.

### P1.5 — K=50 단거리 레짐 (s=10, 25)

ACT 최고점(67.6, K=50 s10)과의 정면 승부 + carry 표 모순(69.5 vs 64.6) 해소.
**TE 없이** 여기서 이기면 최상이다 — 동일 추론 비용에서의 승리라 효율 주장까지 같이 성립한다.

### P2 — `exp4_regime_map` 축 (K × 실행 stride, TE off)

변형 4종(ACT / carry / bimamba / bimos). stride 는 추론 파라미터라 **재학습 없이 eval 플래그만**
바꾸면 된다 → 기존 ckpt 로 커버. 8/18 가설("BiMamba 는 긴 stride, carry 는 짧은 stride")을
하나의 표로 확정하는 게 목적.

### P3 — 순수 BiMamba 학습 (`bimamba_pure`) — **8/24 밤 완료**

아래 "교란 2" 참조. 학습 큐 1순위였고 **K=100 / K=50 / K=150 은 8/25 오전에 150k 로 끝났다**
(`acm2_carry_k150` 도 같이). 남은 건 `acm2_carry_k50`(40k 에서 중단, resume 대상),
`acm2_carry_k20`, `bimamba_pure_k20` 뿐이다.

### 8/25 오전 인벤토리 (실측)

```
act_k20 / act_k50 / act / act_k150            150,000  OK   <- ACT 는 전 K 학습 완료
bimamba_k20 / _k50 / bimamba / _k150          150,000  OK
bimamba_pure / _k50 / _k150                   150,000  OK   <- 8/24 밤 학습
acm2_carry / acm2_carry_k150                  150,000  OK
acm2_carry_k50                                 40,000  PART
acm2_carry_k20 / bimamba_pure_k20                   -  MISS
```

**8/25 오후 확인: act_k10/15 도 이미 학습돼 있었다** (exp3 학습 셀 전부 skip).
따라서 critical 은 `bimamba_pure_k10/15` 2잡만 남는다 → 배정은
**1노드 = 학습(pure k10/15), 2노드 = eval(te_* 500ep 확정 런부터)**.
critical 이 끝나는 내일부터는 전부 eval. (`suggest()` 가 인벤토리에서 자동 계산)

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

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

## 실행 순서

### P1 (최우선) — `exp3_act_te_ksweep.ipynb`

**이 노트북은 아직 한 번도 안 돌았다** (outputs 전부 비어 있음). UBAI가 8/17~8/24 죽어 있어서다.
그래서 K 스윕 표의 `act+TE` 열이 통째로 비어 있다.

| K | act+TE | bimamba+TE | bimamba+carry+TE |
|---|---|---|---|
| 10 | **?** | 75.4 | 74.3 |
| 15 | **?** | 75.6 | 76.4 |
| 20 | **?** | 71.4 | 71.9 |
| 50 | **?** | 64.1 | 63.6 |
| 100 | **?** | 49.3 | 50.2 |
| 150 | **?** | 26.8 | 27.1 |

**이 한 열이 논문 존폐를 가른다.** `bimamba+TE`가 `act+TE`를 이기는 K 구간이 있으면 교수님이 말씀하신
"ACT가 무너지는 지점"이 확보된다. 없으면 3번 분기다.

K=100은 기존 `act` 폴더 재사용이라 학습 스킵, 나머지 5개 K만 학습하면 된다.

### P2 — `exp4_regime_map.ipynb` (신규)

K × 실행 stride 격자, 변형 4종(ACT / carry / bimamba / bimos). **TE off.**
stride는 추론 파라미터라 **재학습 없이 eval 플래그만** 바꾸면 된다 → 기존 ckpt로 전부 커버 가능.

8/18 가설("BiMamba는 긴 stride, carry는 짧은 stride")을 하나의 표로 확정하는 게 목적.

### P3 — 순수 BiMamba 재학습 (P2 결과가 가리키는 K만)

아래 "교란 2" 참조. P2에서 유망한 K가 나오면 그것만.

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

→ `exp4`는 이걸 `bimamba_cpoff`로 **라벨에 명시**해 둔다. 값이 좋게 나오면 그 K만
`use_chunk_pairs=False`로 새로 학습해서 확정한다(P3).

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

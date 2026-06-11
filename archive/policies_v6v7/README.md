# Archived policies (v6/v7 experiments)

v8 (PEC carry / corrected state carryover) 논문 라인업에서 제외된 정책들.
`src/lerobot/policies/`에서 그대로 이동했으며 factory/`policies/__init__.py` 등록은 제거됨.

| 디렉토리 | v6/v7 역할 | 제외 사유 |
|---------|-----------|----------|
| `acm`, `acm2` | B2/B3 Mamba-1/2 baseline | v8은 ACT vs Mamba3 비교만 사용 |
| `act_icpe` | N1 ICPE 대조군 | ICPE 헤드라인 제외 |
| `acm3_icpe` | A1-A3 ICPE ablation | 〃 |
| `acm3_icpe_sscp` | P1/P1cc | 〃 |
| `acm3_icpe_sscp_literal` | P1cc_lit | 〃 (literal carry는 acm3_sscp_literal로 유지) |
| `acm3_bimamba`, `acm3_icpe_sscp_bimamba` | E1a/E1b | E-series 제외 |
| `acm3_self_atten`, `acm3_icpe_sscp_self_atten` | E2a/E2b | 〃 (정신은 state-space ensembling으로 승계) |

복원: 디렉토리를 `src/lerobot/policies/`로 되돌리고 `factory.py`(get_policy_class /
make_policy_config / make_pre_post_processors)와 `policies/__init__.py`에 재등록.
v7 당시 등록 상태는 커밋 `f0ed43e7` 참조.

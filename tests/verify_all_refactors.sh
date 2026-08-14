#!/bin/bash
# Re-verify every refactored variant under the final test settings.
cd /home1/eunji24/lerobot_project/lerobot-bimos
export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-bimos/src
fail=0
for v in acm_bimamba acm_bimamba_gate acm_refiner acm_cross_atten acm_moe acm_self_atten acm_accel_loss; do
    echo "==================== $v ===================="
    python tests/refactor_equivalence.py "$v" || fail=1
done
echo
[ $fail -eq 0 ] && echo "ALL PASSED" || echo "SOME FAILED (see above)"

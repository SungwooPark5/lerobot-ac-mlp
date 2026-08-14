#!/usr/bin/env python
"""Check every acm-family policy name resolves end to end through the factory.

    export PYTHONPATH=/home1/eunji24/lerobot_project/lerobot-bimos/src
    python tests/verify_registration.py

Registering a policy takes four separate edits, in three files, none of which
fail loudly when skipped -- a name can be half-registered and only break when
someone passes it to --policy.type. That is how acm2, acm3 and all seven acm_*
variants ended up unreachable. This checks all of it:

  1. @PreTrainedConfig.register_subclass  -- the name is in the config registry
  2. make_policy_config(name)             -- returns that registry's class
  3. get_policy_class(name)               -- imports and returns the policy class
  4. policy.name / policy.config_class    -- agree with 1-3

Needs a GPU node for (3): importing the modeling modules pulls in mamba_ssm,
whose triton kernels raise at import time without a CUDA driver. Steps 1, 2 and
4's config half run anywhere.

Not checked here: that make_pre_post_processors dispatches correctly, which
relies on isinstance -- every acm_* config subclasses ACMConfig and so takes
acm's branch, while ACM2Config and ACM3Config have their own.
"""

import sys

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_policy_config

NAMES = [
    "acm",
    "acm2",
    "acm3",
    "acm_bimamba",
    "acm_bimamba_gate",
    "acm_refiner",
    "acm_cross_atten",
    "acm_moe",
    "acm_self_atten",
    "acm_accel_loss",
    "acm2_bimamba",
    "acm2_self_atten",
]


def main() -> None:
    failures = []

    for name in NAMES:
        try:
            if name not in PreTrainedConfig.get_known_choices():
                raise AssertionError("missing @PreTrainedConfig.register_subclass")

            cfg = make_policy_config(name)
            registered = PreTrainedConfig.get_choice_class(name)
            if type(cfg) is not registered:
                raise AssertionError(
                    f"make_policy_config returns {type(cfg).__name__}, "
                    f"registry holds {registered.__name__}"
                )

            policy_cls = get_policy_class(name)
            if policy_cls.name != name:
                raise AssertionError(f"policy.name is {policy_cls.name!r}, expected {name!r}")
            if policy_cls.config_class is not registered:
                raise AssertionError(
                    f"policy.config_class is {policy_cls.config_class.__name__}, "
                    f"registry holds {registered.__name__}"
                )

            print(f"  OK    {name:20s} {policy_cls.__name__:24s} {registered.__name__}")
        except Exception as e:  # noqa: BLE001 -- report every name, do not stop at the first
            failures.append((name, e))
            print(f"  FAIL  {name:20s} {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"{len(failures)} of {len(NAMES)} names are not fully registered.")
        sys.exit(1)
    print(f"All {len(NAMES)} names resolve.")


if __name__ == "__main__":
    main()

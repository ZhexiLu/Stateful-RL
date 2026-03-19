"""
Custom reward function for AWM environments in slime.

Vanilla GRPO reward:
  - Completed = 1.0
  - Failed/Incomplete = 0.0
  - Format penalty: -1.0 for invalid tool call format (applied per-step in rollout)

The reward is computed during rollout (in env.compute_reward()),
so this module provides the async_rm interface that slime expects,
simply returning the pre-computed reward.
"""
from __future__ import annotations

from slime.utils.types import Sample


async def async_rm(args, sample: Sample, **kwargs) -> float:
    """
    Return the pre-computed reward from AWM verification.

    The reward is already computed in the rollout function via
    env.compute_reward(), which runs the AWM verification code.
    This function simply returns it.
    """
    if sample.reward is not None:
        return sample.reward
    return 0.0

# ESA dynamic sampling filter for SLIME.
#
# Usage in run script:
#   --dynamic-sampling-filter-path esa_filters.check_esa_group_quality

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

# Minimum fraction of valid (non-infra-failed) samples required in a group.
MIN_VALID_FRACTION = 0.5


def check_esa_group_quality(args, samples: list[Sample], **kwargs):
    """Reject groups with too many infra failures or zero reward variance.

    Combines two checks:
    1. Infra failures: if more than half the group failed, reject and regenerate.
    2. Zero std: if all valid rewards are identical, no GRPO signal — reject.
    """
    n_total = len(samples)
    if n_total == 0:
        return DynamicFilterOutput(keep=False, reason="empty_group")

    # Check infra failures
    n_infra_failed = sum(
        1 for s in samples
        if s.metadata and s.metadata.get("infra_error")
    )
    valid_fraction = (n_total - n_infra_failed) / n_total

    if valid_fraction < MIN_VALID_FRACTION:
        return DynamicFilterOutput(
            keep=False,
            reason=f"infra_failed_{n_infra_failed}/{n_total}",
        )

    # Check reward variance among valid samples
    valid_rewards = [
        s.get_reward_value(args)
        for s in samples
        if not (s.metadata and s.metadata.get("infra_error"))
    ]
    if len(valid_rewards) < 2:
        return DynamicFilterOutput(keep=False, reason="too_few_valid")

    std = torch.tensor(valid_rewards, dtype=torch.float64).std().item()
    if std < 1e-6:
        return DynamicFilterOutput(
            keep=False,
            reason=f"zero_std_{round(valid_rewards[0], 1)}",
        )

    # Check if no sample completed the task — "dead task" with only format bonus noise.
    # These inject reward-hacking signal (more turns = more format bonus) without
    # any task-completion signal for GRPO to learn from.
    n_completed = sum(
        1 for s in samples
        if s.metadata and s.metadata.get("outcome_reward") == 1.0
    )
    if n_completed == 0:
        return DynamicFilterOutput(keep=False, reason="no_completion")

    return DynamicFilterOutput(keep=True)

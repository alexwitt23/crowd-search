"""A file to collate all the available policies and provide an
interface for using them."""

from typing import Any

from crowd_search.policies import ppo_continuous
from crowd_search.policies import ppo_discrete

polcies = {
    "ContinuousPPO": ppo_continuous.ContinuousPPO,
    "DiscretePPO": ppo_discrete.DiscretePPO,
}


def get_policy(policy_key: str) -> Any:
    return polcies[policy_key]


def make_policy(policy_cfg, kwargs):
    policy = get_policy(policy_cfg.get("type"))
    return policy(**policy_cfg, **kwargs)

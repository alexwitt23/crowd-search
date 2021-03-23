"""A file to collate all the available policies and provide an
interface for using them."""

from crowd_search.policies import ppo_continuous
from crowd_search.policies import ppo_discrete

polcies = {
    "ContinuousPPO":ppo_continuous.ContinuousPPO,
    "DiscretePPO": ppo_discrete.DiscretePPO
}


def make_policy(policy_cfg, kwargs):
    policy = polcies[policy_cfg.get("type")]
    return policy(**policy_cfg, **kwargs)

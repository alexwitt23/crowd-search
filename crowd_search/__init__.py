"""Expose the crowd simulation environment."""

from gym.envs.registration import register

register(
    id="CrowdSim-v1", entry_point="crowd_search.crowd_simulation:CrowdSim",
)
register(
    id="GroupCrowdSim-v1", entry_point="crowd_search.groupcrowd_simulation:GroupCrowdSim",
)



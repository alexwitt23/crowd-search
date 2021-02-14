from gym.envs.registration import register

register(
    id="CrowdSim-v1", entry_point="crowd_search.crowd_simulation:CrowdSim",
)

from gym.envs.registration import register

register(
    id="CrowdSim-v0", entry_point="third_party.crowd_sim.envs:CrowdSim",
)

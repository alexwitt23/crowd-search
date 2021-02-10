from setuptools import setup


setup(
    name="crowdnav",
    version="0.0.1",
    packages=[
        "third_party.crowd_sim",
        "third_party.crowd_sim.envs",
        "third_party.crowd_sim.envs.policy",
        "third_party.crowd_sim.envs.utils",
    ],
)

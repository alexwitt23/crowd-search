"""A collection of configurations to control how the crowd moves.

We want to create various different crowds with different variations and levels of a 
few different properties.

1. number of groups in a crowd
2. number of people per group
3. speed of people in a group
4. group direction
    a. there could be multiple separate groups moving towards the same goal

"""

MIN_NUM_GROUPS = 10
MAX_NUM_GROUPS = 10

MIN_NUM_PER_GROUP = 3
MAX_NUM_PER_GROUP = 10

GROUP_RADIUS = 3.0

WORLD_HEIGHT = 50
WORLD_WIDTH = 50


MOTION_PLANNER_CFG = {
    "human_motion_planner": "ORCA",
    "safety-space": 0.0,
    "neighbor-distance": 5.0,
    "max-neighbors": 100,
    "time-horizon": 5.0,
    "time-horizon-obst": 5.0,
    "radius": 0.5,
    "max-speed": 1.0,
    "time-step": 0.25,
}

HUMAN_CFG = {"radius": 0.5, "preferred-velocity": 1.0}

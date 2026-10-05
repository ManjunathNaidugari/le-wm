"""Robot-relative goal geometry. These fields are not motion targets."""
import numpy as np


def pose_and_goal(position, rotation, goal):
    """Y-up world; yaw 0 faces -Z, positive yaw turns left toward -X.

    Habitat AgentState.rotation is numpy-quaternion, not an indexed XYZW array.
    Relative goal is (forward, left, up), in metres.
    """
    from habitat_sim.utils.common import quat_rotate_vector
    forward = quat_rotate_vector(rotation, np.array([0., 0., -1.]))
    heading = float(np.arctan2(-forward[0], -forward[2]))
    local = quat_rotate_vector(rotation.inverse(), np.asarray(goal) - position)
    return heading, np.array([-local[2], -local[0], local[1]], dtype=np.float32)


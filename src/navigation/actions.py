"""Stable action IDs shared by simulator and saved trajectories."""
from enum import IntEnum
from numbers import Integral


class Action(IntEnum):
    FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3


def from_lab_action(action):
    """Habitat-Lab 0.3.3: STOP=0, FORWARD=1, LEFT=2, RIGHT=3."""
    if not isinstance(action, Integral) or isinstance(action, bool) or int(action) not in range(4):
        raise ValueError(f'Unsupported Habitat-Lab navigation action: {action}')
    return (Action.STOP, Action.FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT)[int(action)]


def to_lab_action(action):
    return (1, 2, 3, 0)[int(Action(action))]

"""Obstacle-aware geodesic expert; no learned policy or planner."""
from .actions import Action


def make_follower(habitat_sim, pathfinder, agent, goal_radius):
    return habitat_sim.nav.GreedyGeodesicFollower(
        pathfinder, agent, goal_radius, stop_key=int(Action.STOP),
        forward_key=int(Action.FORWARD), left_key=int(Action.TURN_LEFT),
        right_key=int(Action.TURN_RIGHT))

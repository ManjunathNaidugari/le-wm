"""Execute a supplied official PointNav start/goal without random resampling."""
import numpy as np
from envs.habitat_wrapper import Action, HabitatSmokeEnv


class PointNavEnv(HabitatSmokeEnv):
    def __init__(self, definition, seed=42, resolution=224):
        self.definition = definition
        radius = definition['source_goal_radius']
        super().__init__(definition['scene'], seed, resolution,
                         goal_radius=0.2 if radius is None else radius)

    def reset(self):
        from habitat_sim.utils.common import quat_from_coeffs
        definition = self.definition
        start = np.asarray(definition['start_position'], dtype=np.float32)
        self.goal = np.asarray(definition['goal_position'], dtype=np.float32)
        for name, point in [('start', start), ('goal', self.goal)]:
            if not self.sim.pathfinder.is_navigable(point, max_y_delta=0.1):
                raise RuntimeError(f'Official {name} is not navigable on this navmesh; no snapping or resampling')
        path = self.shortest_path(start, self.goal)
        if path is None:
            raise RuntimeError('Official start and goal are disconnected; no replacement episode')
        self.initial_path = np.asarray(path.points, dtype=np.float32)
        self.initial_distance = float(path.geodesic_distance)
        state = self.hs.AgentState()
        state.position = start
        state.rotation = quat_from_coeffs(definition['start_rotation_xyzw'])
        self.agent.set_state(state)
        self.follower = self.hs.nav.GreedyGeodesicFollower(
            self.sim.pathfinder, self.agent, self.goal_radius,
            stop_key=int(Action.STOP), forward_key=int(Action.FORWARD),
            left_key=int(Action.TURN_LEFT), right_key=int(Action.TURN_RIGHT))
        self.done = False
        return self.snapshot(self.sim.get_sensor_observations())

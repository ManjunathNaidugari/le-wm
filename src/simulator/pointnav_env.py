"""Execute a supplied official PointNav start/goal without random resampling."""
from dataclasses import replace
import numpy as np
from jepa_navigation.utils.config import NavigationConfig
from jepa_navigation.navigation.expert import make_follower
from jepa_navigation.simulator.habitat_env import Action, HabitatSmokeEnv


class PointNavEnv(HabitatSmokeEnv):
    def __init__(self, definition, seed=42, resolution=224, *,
                 simulator_config=None, navigation_config=None):
        self.definition = definition
        radius = definition['source_goal_radius']
        navigation_config = navigation_config or NavigationConfig()
        if radius is not None:
            navigation_config = replace(navigation_config, goal_radius_m=radius)
        super().__init__(definition['scene'], seed, resolution,
                         simulator_config=simulator_config, navigation_config=navigation_config)

    def reset(self):
        from habitat_sim.utils.common import quat_from_coeffs
        definition = self.definition
        start = np.asarray(definition['start_position'], dtype=np.float32)
        self.goal = np.asarray(definition['goal_position'], dtype=np.float32)
        for name, point in [('start', start), ('goal', self.goal)]:
            if not self.sim.pathfinder.is_navigable(point, max_y_delta=getattr(self, 'navigation_config', NavigationConfig()).max_y_delta_m):
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
        self.follower = make_follower(self.hs, self.sim.pathfinder, self.agent, self.goal_radius)
        self.done = False
        return self.snapshot(self.sim.get_sensor_observations())

"""Habitat-Lab task/metrics around Habitat-Sim, using the existing recorder API."""
import json
from pathlib import Path

import numpy as np

from jepa_navigation.navigation.actions import Action, from_lab_action, to_lab_action
from jepa_navigation.navigation.coordinates import pose_and_goal
from jepa_navigation.simulator.habitat_env import require_habitat


def require_lab():
    hs = require_habitat()
    try:
        import habitat
    except (ImportError, OSError) as exc:
        raise RuntimeError('Habitat-Lab 0.3.3 required; use environment-gibson-pilot.yml') from exc
    if habitat.__version__ != '0.3.3':
        raise RuntimeError(f'Expected Habitat-Lab 0.3.3, found {habitat.__version__}')
    return hs, habitat


def build_lab_config(habitat, sim, nav, radius):
    from habitat.config import read_write
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig, TopDownMapMeasurementConfig,
    )
    if not float(nav.turn_degrees).is_integer():
        raise ValueError('Habitat-Lab 0.3.3 requires integer turn_degrees; refusing to round')
    if not float(sim.hfov_degrees).is_integer():
        raise ValueError('Habitat-Lab 0.3.3 requires integer hfov_degrees; refusing to round')
    config = habitat.get_config('benchmark/nav/pointnav/pointnav_gibson.yaml')
    with read_write(config):
        h = config.habitat
        h.seed = nav.seed
        h.dataset.split = 'train'
        h.dataset.data_path = nav.episode_data
        h.dataset.scenes_dir = nav.scene_data_dir
        h.environment.max_episode_steps = nav.max_steps
        h.environment.max_episode_seconds = 0
        h.environment.iterator_options.shuffle = False
        h.environment.iterator_options.cycle = True
        h.simulator.forward_step_size = nav.forward_step_m
        h.simulator.turn_angle = int(nav.turn_degrees)
        h.simulator.habitat_sim_v0.enable_physics = False
        agent = h.simulator.agents.main_agent
        agent.height, agent.radius = sim.agent_height_m, sim.agent_radius_m
        rgb = agent.sim_sensors.rgb_sensor
        rgb.height = rgb.width = sim.resolution
        rgb.position = [0., sim.camera_height_m, 0.]
        rgb.hfov = int(sim.hfov_degrees)
        agent.sim_sensors = {'rgb_sensor': rgb}
        h.task.lab_sensors = {}
        h.task.measurements.success.success_distance = radius
        h.task.measurements.distance_to_goal.distance_to = 'POINT'
        topdown = TopDownMapMeasurementConfig(map_resolution=512, max_episode_steps=nav.max_steps)
        topdown.fog_of_war.draw = False
        topdown.draw_goal_aabbs = False
        topdown.draw_view_points = False
        h.task.measurements.top_down_map = topdown
        h.task.measurements.collisions = CollisionsMeasurementConfig()
    return config


class LabPointNavEnv:
    """One supplied episode. No random start/goal sampling, snapping or replacement."""
    def __init__(self, definition, simulator_config, navigation_config):
        hs, habitat = require_lab()
        from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
        from habitat.config import read_write
        self.hs, self.definition = hs, definition
        self.navigation_config = navigation_config
        self.goal_radius = definition['source_goal_radius'] or navigation_config.goal_radius_m
        if not Path(definition['scene']).is_file():
            raise FileNotFoundError(f"Gibson scene missing: {definition['scene']}")
        self.goal = np.asarray(definition['goal_position'], dtype=np.float32)
        dataset = PointNavDatasetV1()
        # Keep official metadata/shortest_paths too. Only relocate the scene path.
        raw = dict(definition['source_definition'], scene_id=definition['scene'])
        dataset.from_json(json.dumps({'episodes': [raw]}))
        config = build_lab_config(habitat, simulator_config, navigation_config, self.goal_radius)
        with read_write(config):
            config.habitat.dataset.content_scenes = [Path(definition['source_scene_id']).stem]
            config.habitat.simulator.scene = definition['scene']
        self.config = config
        self.env = habitat.Env(config=config, dataset=dataset)
        self.sim = self.env.sim
        try:
            self.env.seed(navigation_config.seed)
        except Exception:
            self.env.close()
            raise
        self.done = False

    def reset(self):
        from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
        from habitat_sim.utils.common import quat_to_coeffs
        from omegaconf import OmegaConf
        obs = self.env.reset()
        # Lab overwrites simulator scene/start fields during reset; save that effective config.
        self.effective_lab_config = OmegaConf.to_container(self.config, resolve=True)
        state = self.sim.get_agent_state()
        if not np.allclose(state.position, self.definition['start_position'], atol=1e-5, rtol=0):
            raise RuntimeError('Habitat-Lab did not apply the official start position')
        rotation = np.asarray(quat_to_coeffs(state.rotation))
        expected = np.asarray(self.definition['start_rotation_xyzw'])
        if min(np.max(np.abs(rotation - expected)), np.max(np.abs(rotation + expected))) > 1e-5:
            raise RuntimeError('Habitat-Lab did not apply the official start rotation')
        np.testing.assert_allclose(self.env.current_episode.goals[0].position, self.goal, atol=1e-5)
        for name, point in [('start', state.position), ('goal', self.goal)]:
            if not self.sim.pathfinder.is_navigable(point, max_y_delta=self.navigation_config.max_y_delta_m):
                raise RuntimeError(f'Official {name} is not navigable; no snapping or resampling')
        path = self.hs.ShortestPath()
        path.requested_start, path.requested_end = state.position, self.goal
        if not self.sim.pathfinder.find_path(path) or not np.isfinite(path.geodesic_distance):
            raise RuntimeError('Official start and goal are disconnected; no replacement episode')
        self.initial_path = np.asarray(path.points, dtype=np.float32)
        self.initial_distance = float(path.geodesic_distance)
        # Explicitly expose follower failure; Lab's default would convert it to STOP.
        self.follower = ShortestPathFollower(self.sim, self.goal_radius, False, stop_on_error=False)
        self.done = False
        return self.snapshot(obs)

    def snapshot(self, obs):
        from habitat.utils.visualizations import maps
        from habitat_sim.utils.common import quat_to_coeffs
        state = self.sim.get_agent_state()
        position = np.array(state.position, dtype=np.float32, copy=True)
        heading, relative = pose_and_goal(position, state.rotation, self.goal)
        rgb = np.array(obs['rgb'][..., :3], copy=True)
        topdown = maps.colorize_draw_agent_and_fit_to_height(
            self.env.get_metrics()['top_down_map'], rgb.shape[0])
        frame = np.concatenate((rgb, topdown), axis=1)
        # H264 yuv420 needs even dimensions; pad the map edge, preserve raw RGB.
        if frame.shape[1] % 2:
            frame = np.pad(frame, ((0, 0), (0, 1), (0, 0)), mode='edge')
        return dict(rgb=rgb, position=position, heading=heading, relative_goal=relative,
                    rotation_xyzw=np.asarray(quat_to_coeffs(state.rotation), dtype=np.float32),
                    visualization=frame)

    def expert_action(self):
        action = self.follower.get_next_action(self.goal)
        if action is None:
            raise RuntimeError('Follower returned no action; episode was not replaced')
        return from_lab_action(action)

    def step(self, action):
        if self.done or self.env.episode_over:
            raise RuntimeError('Episode already terminated')
        action = Action(action)
        # Task action names avoid relying on Hydra dictionary insertion order.
        names = ('stop', 'move_forward', 'turn_left', 'turn_right')
        obs = self.env.step(names[to_lab_action(action)])
        self.done = self.env.episode_over
        collision = action != Action.STOP and bool(self.sim.previous_step_collided)
        return self.snapshot(obs), collision

    def distance(self):
        return float(self.env.get_metrics()['distance_to_goal'])

    def reference_metrics(self):
        values = self.env.get_metrics()
        return {key: float(values[key]) for key in ('success', 'spl', 'distance_to_goal')}

    def close(self):
        self.env.close()

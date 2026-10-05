"""Small Habitat-Sim 0.3.3 adapter; no Habitat-Lab or Gym API mixing."""
from pathlib import Path
import numpy as np
from jepa_navigation.navigation.actions import Action
from jepa_navigation.navigation.coordinates import pose_and_goal
from jepa_navigation.navigation.expert import make_follower
from jepa_navigation.simulator.sensors import rgb_sensor
from jepa_navigation.utils.config import SimulatorConfig, NavigationConfig




def require_habitat():
    try:
        import habitat_sim
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Habitat-Sim 0.3.3 is required. Create the environment from "
            "environment-habitat.yml; see README.md. No synthetic data is generated."
        ) from exc
    if habitat_sim.__version__ != "0.3.3":
        raise RuntimeError(f"Expected Habitat-Sim 0.3.3, found {habitat_sim.__version__}")
    return habitat_sim




class HabitatSmokeEnv:
    def __init__(self, scene, seed=42, resolution=224, goal_radius=0.2, *,
                 simulator_config=None, navigation_config=None):
        self.simulator_config = simulator_config or SimulatorConfig(resolution=resolution)
        self.navigation_config = navigation_config or NavigationConfig(goal_radius_m=goal_radius)
        hs = require_habitat()
        if not Path(scene).is_file():
            raise FileNotFoundError(
                f"Scene not found: {scene}. Supply the matching external scene assets."
            )
        self.hs, self.goal_radius = hs, self.navigation_config.goal_radius_m
        cfg = hs.SimulatorConfiguration()
        cfg.scene_id = str(Path(scene).resolve())
        cfg.enable_physics = False
        sensor = rgb_sensor(hs, self.simulator_config)
        agent_cfg = hs.agent.AgentConfiguration()
        agent_cfg.height = self.simulator_config.agent_height_m
        agent_cfg.radius = self.simulator_config.agent_radius_m
        agent_cfg.sensor_specifications = [sensor]
        agent_cfg.action_space = {
            int(key): hs.agent.ActionSpec(name, hs.agent.ActuationSpec(amount=amount))
            for key, name, amount in [(Action.FORWARD, "move_forward", self.navigation_config.forward_step_m),
                                      (Action.TURN_LEFT, "turn_left", self.navigation_config.turn_degrees),
                                      (Action.TURN_RIGHT, "turn_right", self.navigation_config.turn_degrees)]
        }
        self.sim = hs.Simulator(hs.Configuration(cfg, [agent_cfg]))
        try:
            self.sim.seed(seed)
            # Rebuild even if the scene ships a mesh: match the actual agent dimensions.
            settings = hs.NavMeshSettings()
            settings.set_defaults()
            settings.agent_height, settings.agent_radius = agent_cfg.height, agent_cfg.radius
            if not self.sim.recompute_navmesh(self.sim.pathfinder, settings):
                raise RuntimeError("Could not build a navigable navmesh for this scene")
            self.sim.pathfinder.seed(seed)
            self.agent = self.sim.initialize_agent(0)
            self.seed = seed
        except Exception:
            self.close()
            raise

    def shortest_path(self, start, goal):
        path = self.hs.ShortestPath()
        path.requested_start, path.requested_end = start, goal
        found = self.sim.pathfinder.find_path(path)
        return path if found and np.isfinite(path.geodesic_distance) else None

    def reset(self):
        from habitat_sim.utils.common import quat_from_angle_axis
        for _ in range(1000):
            start = self.sim.pathfinder.get_random_navigable_point()
            goal = self.sim.pathfinder.get_random_navigable_point()
            path = self.shortest_path(start, goal)
            if path is not None and 2.0 <= path.geodesic_distance <= 10.0:
                break
        else:
            raise RuntimeError("Could not sample a connected start/goal pair 2–10m apart")
        self.goal = np.array(goal, dtype=np.float32)
        self.initial_path = np.array(path.points, dtype=np.float32)
        self.initial_distance = float(path.geodesic_distance)
        state = self.hs.AgentState()
        state.position = start
        state.rotation = quat_from_angle_axis(
            np.random.default_rng(self.seed).uniform(-np.pi, np.pi), np.array([0., 1., 0.]))
        self.agent.set_state(state)
        self.follower = make_follower(self.hs, self.sim.pathfinder, self.agent, self.goal_radius)
        self.done = False
        return self.snapshot(self.sim.get_sensor_observations())

    def snapshot(self, observations):
        state = self.agent.get_state()
        position = np.array(state.position, dtype=np.float32, copy=True)
        heading, relative = pose_and_goal(position, state.rotation, self.goal)
        return {"rgb": np.array(observations["rgb"][..., :3], copy=True),
                "position": position, "heading": heading, "relative_goal": relative}

    def distance(self):
        path = self.shortest_path(self.agent.get_state().position, self.goal)
        return float(path.geodesic_distance) if path is not None else float("inf")

    def expert_action(self):
        return Action(self.follower.next_action_along(self.goal))

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode already stopped")
        action = Action(action)
        if action == Action.STOP:
            # STOP is a terminal no-op, not a Habitat-Sim movement control.
            self.done = True
            obs, collision = self.sim.get_sensor_observations(), False
        else:
            obs = self.sim.step(int(action))
            collision = bool(obs["collided"])
        return self.snapshot(obs), collision

    def close(self):
        self.sim.close()

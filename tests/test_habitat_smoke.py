"""Unit doubles test contracts only; they are never used by the demo."""
import math
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import numpy as np
import torch
from jepa_navigation.data.habitat_collector import collect_episode
from jepa_navigation.data.habitat_dataset import HabitatTrajectoryDataset, validate_trajectory
from jepa_navigation.simulator.habitat_env import Action, HabitatSmokeEnv, pose_and_goal, require_habitat


class FakeEnv:
    goal_radius = .2
    goal = np.array([0., 0., -1.], dtype=np.float32)
    initial_path = np.array([[0., 0., 0.], [0., 0., -1.]], dtype=np.float32)
    initial_distance = 1.

    def state(self):
        return dict(rgb=np.full((4, 4, 3), self.t, np.uint8),
                    position=np.array([0., 0., -float(self.t)], np.float32),
                    heading=0., relative_goal=np.array([1. - self.t, 0., 0.], np.float32))

    def reset(self):
        self.t = 0
        return self.state()

    def expert_action(self):
        return Action.FORWARD if self.t == 0 else Action.STOP

    def step(self, action):
        if action != Action.STOP:
            self.t += 1
        return self.state(), action == Action.FORWARD

    def distance(self):
        return abs(1. - self.t)


class SmokeTests(unittest.TestCase):
    def test_alignment_and_roundtrip(self):
        ep = collect_episode(FakeEnv())
        validate_trajectory(ep)
        self.assertEqual(ep['actions'].tolist(), [0, 3])
        self.assertEqual(ep['observations'][:, 0, 0, 0].tolist(), [0, 1, 1])
        self.assertEqual(ep['positions'][:, 2].tolist(), [0, -1, -1])
        self.assertEqual(ep['relative_goals'][:, 0].tolist(), [1, 0, 0])
        self.assertEqual(ep['collisions'].tolist(), [True, False])
        self.assertTrue(ep['metrics']['success'])
        self.assertEqual(ep['metrics']['path_length'], 1.)
        with tempfile.TemporaryDirectory() as tmp:
            torch.save(ep, Path(tmp) / 'episode_0001.pt')
            ds = HabitatTrajectoryDataset(tmp)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]['observations'][0, 0, 0, 0], 0)
            self.assertEqual(ds[0]['next_observations'][0, 0, 0, 0], 1)
            self.assertEqual(len(HabitatTrajectoryDataset(tmp, num_steps=2)), 1)

    def test_max_steps_not_success_without_stop(self):
        ep = collect_episode(FakeEnv(), max_steps=1)
        self.assertFalse(ep['metrics']['success'])
        self.assertEqual(ep['metrics']['termination'], 'max_steps')
        validate_trajectory(ep)

    def test_follower_failure_preserves_initial_state(self):
        class GreedyFollowerError(Exception):
            pass
        errors = ModuleType('habitat_sim.errors')
        errors.GreedyFollowerError = GreedyFollowerError
        env = FakeEnv()
        env.expert_action = lambda: (_ for _ in ()).throw(GreedyFollowerError('no route'))
        with patch.dict('sys.modules', {'habitat_sim.errors': errors}):
            ep = collect_episode(env)
        validate_trajectory(ep)
        self.assertEqual(ep['metrics']['termination'], 'follower_error')
        self.assertFalse(ep['metrics']['success'])
        self.assertEqual(len(ep['observations']), 1)
        self.assertEqual(len(ep['actions']), 0)

    def test_old_or_misaligned_data_rejected(self):
        with self.assertRaises(ValueError):
            validate_trajectory({})
        ep = collect_episode(FakeEnv())
        ep['observations'] = ep['observations'][:-1]
        with self.assertRaisesRegex(ValueError, 'T\\+1'):
            validate_trajectory(ep)

    def test_missing_habitat_is_error(self):
        with patch.dict('sys.modules', {'habitat_sim': None}):
            with self.assertRaisesRegex(RuntimeError, 'No synthetic data'):
                require_habitat()

    def test_stop_never_calls_sim_step(self):
        env = HabitatSmokeEnv.__new__(HabitatSmokeEnv)
        env.done = False
        class Sim:
            def get_sensor_observations(self):
                return {'rgb': 'real sensor data'}
        env.sim = Sim()
        env.snapshot = lambda obs: obs
        obs, collision = env.step(Action.STOP)
        self.assertEqual(obs['rgb'], 'real sensor data')
        self.assertFalse(collision)
        with self.assertRaises(RuntimeError):
            env.step(Action.FORWARD)

    def test_heading_conventions(self):
        # Independent rotation matrices exercise forward/left signs without Habitat.
        class Rotation:
            def __init__(self, yaw):
                self.yaw = yaw
            def inverse(self):
                return Rotation(-self.yaw)
        def rotate(q, v):
            c, s = math.cos(q.yaw), math.sin(q.yaw)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ v
        common = ModuleType('habitat_sim.utils.common')
        common.quat_rotate_vector = rotate
        with patch.dict('sys.modules', {'habitat_sim.utils.common': common}):
            for yaw in (0., math.pi / 2, -math.pi / 2, math.pi):
                goal = rotate(Rotation(yaw), np.array([-2., 3., -4.]))
                heading, relative = pose_and_goal(np.zeros(3), Rotation(yaw), goal)
                self.assertAlmostEqual(heading, yaw)
                np.testing.assert_allclose(relative, [4., 2., 3.], atol=1e-6)

    @unittest.skipUnless(__import__('importlib').util.find_spec('habitat_sim'), 'Habitat-Sim not installed')
    def test_real_quaternion_api(self):
        from habitat_sim.utils.common import quat_from_angle_axis
        rotation = quat_from_angle_axis(math.pi / 2, np.array([0., 1., 0.]))
        heading, relative = pose_and_goal(np.zeros(3), rotation, np.array([-2., 0., 0.]))
        self.assertAlmostEqual(heading, math.pi / 2)
        np.testing.assert_allclose(relative, [2., 0., 0.], atol=1e-6)


if __name__ == '__main__':
    unittest.main()

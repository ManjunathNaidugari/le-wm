"""Official-format fixtures are unit data, never substituted in production."""
import gzip
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data.pointnav import load_pointnav, parse_episode
from data.pointnav_collection import collect_pointnav
from data.habitat_dataset_collection import validate_dataset
from datasets.habitat_dataset import HabitatTrajectoryDataset
from envs.pointnav_wrapper import PointNavEnv
from test_habitat_smoke import FakeEnv


def raw_episode(scene='gibson/A.glb', episode_id='0'):
    return dict(episode_id=episode_id, scene_id='data/scene_datasets/' + scene,
                start_position=[0., 0., 0.], start_rotation=[0., 0., 0., 1.],
                goals=[dict(position=[0., 0., -1.], radius=None)])


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt') as stream:
        json.dump(payload, stream)


class RecordedEnv(FakeEnv):
    closed = 0
    def __init__(self, definition, seed, resolution):
        self.goal = np.array(definition['goal_position'], dtype=np.float32)
    def close(self):
        RecordedEnv.closed += 1


class PointNavTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scenes = self.root / 'scenes'
        (self.scenes / 'gibson').mkdir(parents=True)
        for name in ('A', 'B'):
            (self.scenes / 'gibson' / f'{name}.glb').touch()  # parser fixture only

    def test_monolithic_parsing_preserves_pose(self):
        path = self.root / 'val.json'
        raw = raw_episode()
        raw['start_rotation'] = [0., np.sin(np.pi/4), 0., np.cos(np.pi/4)]
        write_json(path, dict(episodes=[raw]))
        result = load_pointnav(path, self.scenes, 1)[0]
        self.assertEqual(result['start_rotation_xyzw'], raw['start_rotation'])
        self.assertEqual(result['start_position'], raw['start_position'])
        self.assertEqual(result['goal_position'], raw['goals'][0]['position'])
        self.assertEqual(result['source_episode_id'], '0')
        self.assertEqual(len(result['source_episode_sha256']), 64)

    def test_shards_deterministic_scene_variation(self):
        index = self.root / 'val.json.gz'
        write_json(index, dict(episodes=[], content_scenes_path='{data_path}/content/{scene}.json.gz'))
        for scene in ('A', 'B'):
            write_json(self.root / 'content' / f'{scene}.json.gz',
                       dict(episodes=[raw_episode(f'gibson/{scene}.glb', str(i)) for i in range(10)]))
        selected = load_pointnav(index, self.scenes, 4, seed=42)
        self.assertEqual(selected, load_pointnav(index, self.scenes, 4, seed=42))
        self.assertNotEqual(selected, load_pointnav(index, self.scenes, 4, seed=43))
        self.assertEqual(len({e['scene'] for e in selected[:2]}), 2)
        self.assertEqual(len({(e['scene'], e['source_episode_id']) for e in selected}), 4)

    def test_invalid_definitions_no_replacements(self):
        mutations = [lambda r: r.update(start_rotation=[0, 0, 0, 0]),
                     lambda r: r.update(start_position=[float('nan'), 0, 0]),
                     lambda r: r.update(goals=[]),
                     lambda r: r['goals'].append(r['goals'][0]),
                     lambda r: r.update(scene_id='../escape.glb'),
                     lambda r: r.update(scene_id='/tmp/absolute.glb'),
                     lambda r: r['goals'][0].update(radius=-1)]
        for mutation in mutations:
            raw = raw_episode()
            mutation(raw)
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_episode(raw, self.scenes, 'fixture', 'hash')
        raw = raw_episode('gibson/missing.glb')
        with self.assertRaises(FileNotFoundError):
            parse_episode(raw, self.scenes, 'fixture', 'hash')
        path = self.root / 'empty.json.gz'
        write_json(path, dict(episodes=[]))
        with self.assertRaisesRegex(ValueError, 'found only'):
            load_pointnav(path, self.scenes, 1)

    def test_collection_metadata_alignment_and_loader(self):
        definitions = [parse_episode(raw_episode(f'gibson/{name}.glb'), self.scenes, 'fixture', 'hash')
                       for name in ('A', 'B')]
        RecordedEnv.closed = 0
        with patch('data.pointnav_collection.PointNavEnv', RecordedEnv):
            first = collect_pointnav(definitions, self.root / 'out', max_steps=3)
            second = collect_pointnav(definitions, self.root / 'repeat', max_steps=3)
        self.assertEqual(first, second)
        self.assertEqual(RecordedEnv.closed, 4)
        self.assertEqual(validate_dataset(self.root / 'out'), first)
        ds = HabitatTrajectoryDataset(self.root / 'out')
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds[0]['observations'].shape[0], 1)
        self.assertEqual(ds[0]['next_positions'][0, 2], -1.)
        episode = torch.load(self.root / 'out/episode_000000.pt', weights_only=True)
        meta = episode['metadata']
        self.assertEqual(meta['start_position'], definitions[0]['start_position'])
        self.assertEqual(meta['camera']['resolution'], [224, 224])
        self.assertEqual(meta['source_episode_id'], '0')
        self.assertEqual(episode['observations'].shape[0], len(episode['actions'])+1)

    def test_reset_uses_exact_official_pose_without_sampling(self):
        raw = raw_episode()
        raw['start_rotation'] = [0., np.sin(np.pi/4), 0., np.cos(np.pi/4)]
        definition = parse_episode(raw, self.scenes, 'fixture', 'hash')
        env = PointNavEnv.__new__(PointNavEnv)
        env.definition = definition
        env.goal_radius = .2
        states = []
        follower_calls = []
        env.agent = SimpleNamespace(set_state=states.append)
        env.sim = SimpleNamespace(pathfinder=SimpleNamespace(is_navigable=lambda point, max_y_delta: True),
                                  get_sensor_observations=lambda: {'rgb': 'real sensor'})
        env.hs = SimpleNamespace(AgentState=SimpleNamespace,
                                 nav=SimpleNamespace(GreedyGeodesicFollower=lambda *a, **kw: follower_calls.append(kw)))
        env.shortest_path = lambda a, b: SimpleNamespace(points=[a, b], geodesic_distance=1.)
        env.snapshot = lambda observations: observations
        common = ModuleType('habitat_sim.utils.common')
        common.quat_from_coeffs = lambda xyzw: ('quaternion', tuple(xyzw))
        with patch.dict('sys.modules', {'habitat_sim.utils.common': common}):
            self.assertEqual(env.reset()['rgb'], 'real sensor')
        np.testing.assert_array_equal(states[0].position, definition['start_position'])
        self.assertEqual(states[0].rotation, ('quaternion', tuple(raw['start_rotation'])))
        self.assertEqual(follower_calls[0]['stop_key'], 3)
        env.sim.pathfinder.is_navigable = lambda point, max_y_delta: False
        with patch.dict('sys.modules', {'habitat_sim.utils.common': common}), self.assertRaisesRegex(RuntimeError, 'no snapping'):
            env.reset()

    @unittest.skipUnless(importlib.util.find_spec('habitat_sim') and
                         os.environ.get('HABITAT_GIBSON_SCENE_DIR') and
                         os.environ.get('HABITAT_GIBSON_EPISODES'),
                         'Requires Habitat-Sim and explicit HABITAT_GIBSON_SCENE_DIR/HABITAT_GIBSON_EPISODES assets')
    def test_real_gibson_episode(self):
        definitions = load_pointnav(os.environ['HABITAT_GIBSON_EPISODES'],
                                   os.environ['HABITAT_GIBSON_SCENE_DIR'], 1)
        collect_pointnav(definitions, self.root / 'real', max_steps=500)
        validate_dataset(self.root / 'real', check_rgb_quality=False)
        episode = torch.load(self.root / 'real/episode_000000.pt', weights_only=True)
        np.testing.assert_allclose(episode['positions'][0], definitions[0]['start_position'], atol=1e-5)
        np.testing.assert_allclose(episode['goal_position'], definitions[0]['goal_position'], atol=1e-5)
        self.assertGreater(len(episode['actions']), 0)

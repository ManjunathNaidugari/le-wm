"""Focused unit tests. Fixture rollouts are never real-data acceptance evidence."""
from dataclasses import replace
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from jepa_navigation.data.habitat_collector import collect_episode
from jepa_navigation.data.pointnav import load_pointnav
from jepa_navigation.data.gibson_pilot import collect_pilot, compare_episodes, validate_pilot, validate_pilot_run
from jepa_navigation.navigation.actions import Action, from_lab_action, to_lab_action
from jepa_navigation.utils.config import SimulatorConfig, NavigationConfig, effective_settings
from jepa_navigation.utils.pilot_cli import main
from jepa_navigation.utils.video import export_video
from test_pointnav import raw_episode, write_json


class FixtureEnv:
    goal_radius = 0.2
    goal = np.array([0., 0., -1.], dtype=np.float32)
    initial_path = np.array([[0., 0., 0.], [0., 0., -1.]], dtype=np.float32)
    initial_distance = 1.

    def state(self):
        rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        position = np.array([0., 0., -float(self.moved)], dtype=np.float32)
        return dict(rgb=rgb, position=position, heading=0., rotation_xyzw=np.array([0., 0., 0., 1.], dtype=np.float32),
                    relative_goal=np.array([1.-float(self.moved), 0., 0.], dtype=np.float32),
                    visualization=np.concatenate([rgb, rgb], axis=1))

    def reset(self):
        self.moved = False
        self.stopped = False
        return self.state()

    def expert_action(self):
        return Action.STOP if self.moved else Action.FORWARD

    def step(self, action):
        if action == Action.FORWARD:
            self.moved = True
        if action == Action.STOP:
            self.stopped = True
        return self.state(), False

    def distance(self):
        return 0. if self.moved else 1.

    def reference_metrics(self):
        return dict(success=float(self.stopped), spl=float(self.stopped), distance_to_goal=self.distance())


def fixture_episode(definition, sim, nav):
    episode = collect_episode(FixtureEnv(), nav.max_steps)
    episode['metadata'] = dict(definition, seed=nav.seed, episode_id=0, goal_radius_m=0.2,
                               forward_step_m=nav.forward_step_m, turn_degrees=nav.turn_degrees,
                               habitat_sim_version='unit-fixture', habitat_lab_version='unit-fixture',
                               effective_settings=effective_settings(sim, nav), scene_sha256='fixture',
                               packages={}, platform='fixture', controller='fixture')
    return episode


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scenes = self.root / 'scenes'
        (self.scenes / 'gibson').mkdir(parents=True)
        (self.scenes / 'gibson/A.glb').write_bytes(b'parser fixture, never a real mesh')
        self.index = self.root / 'train/train.json.gz'
        write_json(self.index, dict(episodes=[], content_scenes_path='{data_path}/content/{scene}.json.gz'))
        write_json(self.index.parent / 'content/A.json.gz', dict(episodes=[raw_episode(episode_id=str(i)) for i in range(5)]))
        # Unrelated shard is unreadable and its mesh is absent: selecting A must work.
        (self.index.parent / 'content/B.json.gz').write_bytes(b'not gzip')
        self.definition = load_pointnav(self.index, self.scenes, 1, scenes=['A'], episode_ids=['0'])[0]
        self.sim, self.nav = SimulatorConfig(resolution=4), NavigationConfig()

    def test_subset_exact_id_and_no_unrelated_files(self):
        self.assertEqual(self.definition['source_episode_id'], '0')
        self.assertEqual(self.definition['source_definition'], raw_episode())
        with self.assertRaises(ValueError):
            load_pointnav(self.index, self.scenes, 1, scenes=['../A'])
        with self.assertRaises(ValueError):
            load_pointnav(self.index, self.scenes, 1, scenes=['A'], episode_ids=['99'])

    def test_missing_selected_mesh_can_be_accounted_for(self):
        (self.scenes / 'gibson/A.glb').unlink()
        definition = load_pointnav(self.index, self.scenes, 1, scenes=['A'], require_scenes=False)[0]
        result = collect_pilot([definition], self.root / 'missing', self.sim, self.nav)
        self.assertEqual(result['episodes'][0]['status'], 'error')
        self.assertIn('FileNotFoundError', result['episodes'][0]['error'])
        validate_pilot_run(self.root / 'missing')

    def test_action_mapping_all_four(self):
        self.assertEqual([int(from_lab_action(i)) for i in range(4)], [3, 0, 1, 2])
        for action in Action:
            self.assertEqual(from_lab_action(to_lab_action(action)), action)
        with self.assertRaises(ValueError):
            from_lab_action(4)
        with self.assertRaises(ValueError):
            from_lab_action(1.5)

    def test_controller_alignment_and_stop(self):
        calls = []
        def controller(state):
            calls.append(state)
            return Action.FORWARD if len(calls) == 1 else Action.STOP
        episode = collect_episode(FixtureEnv(), controller=controller)
        self.assertEqual(episode['actions'].tolist(), [0, 3])
        self.assertEqual(len(episode['observations']), 3)
        self.assertEqual(len(episode['rotations_xyzw']), 3)
        self.assertEqual(len(calls), 2)

    def test_validation_rejects_changed_pose_goal_and_stop(self):
        original = fixture_episode(self.definition, self.sim, self.nav)
        validate_pilot(original)
        for field, index in [('positions', (0, 0)), ('rotations_xyzw', (0, 0)),
                             ('relative_goals', (0, 0)), ('observations', (-1, 0, 0, 0))]:
            corrupted = copy.deepcopy(original)
            corrupted[field][index] += 1
            with self.assertRaises((ValueError, AssertionError)):
                validate_pilot(corrupted)
        boundary = copy.deepcopy(original)
        boundary['metrics']['final_distance'] = 0.2
        with self.assertRaisesRegex(ValueError, 'strict'):
            validate_pilot(boundary)

    def test_manifest_continues_after_failure_and_refuses_overwrite(self):
        second = copy.deepcopy(self.definition)
        second['source_episode_id'] = '1'
        def runner(definition, sim, nav):
            if definition['source_episode_id'] == '0':
                raise RuntimeError('Disconnected official episode')
            return fixture_episode(definition, sim, nav)
        root = self.root / 'run'
        manifest = collect_pilot([self.definition, second], root, self.sim, self.nav, runner=runner)
        self.assertEqual([e['status'] for e in manifest['episodes']], ['error', 'finished'])
        self.assertEqual(manifest['summary']['attempted'], 2)
        self.assertEqual(manifest['summary']['successes'], 1)
        self.assertEqual(manifest['real_data_acceptance'], 'pending')
        validate_pilot_run(root)
        with self.assertRaises(FileExistsError):
            collect_pilot([self.definition], root, self.sim, self.nav, runner=runner)
        with self.assertRaises(ValueError):
            collect_pilot([self.definition] * 21, self.root / 'large', self.sim, self.nav)

    def test_video_failure_keeps_trajectory(self):
        root = self.root / 'encode_failure'
        with patch('jepa_navigation.data.gibson_pilot.export_video', side_effect=RuntimeError('encoder unavailable')):
            result = collect_pilot([self.definition], root, self.sim, self.nav, runner=fixture_episode)
        self.assertEqual(result['episodes'][0]['status'], 'video_error')
        self.assertTrue((root / 'episode_000000.pt').exists())
        validate_pilot_run(root)

    def test_executed_timeout_is_saved_as_failure(self):
        root = self.root / 'timeout'
        result = collect_pilot([self.definition], root, self.sim, replace(self.nav, max_steps=1), runner=fixture_episode)
        entry = result['episodes'][0]
        self.assertEqual(entry['status'], 'finished')
        self.assertEqual(entry['termination'], 'max_steps')
        self.assertFalse(entry['success'])
        validate_pilot_run(root)

    def test_manifest_corruption_is_rejected(self):
        root = self.root / 'corrupt'
        result = collect_pilot([self.definition], root, self.sim, self.nav, runner=fixture_episode)
        result['episodes'][0]['steps'] += 1
        (root / 'manifest.json').write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, 'match trajectory'):
            validate_pilot_run(root)

    def test_repeatability_detects_rgb_change_and_settings_mismatch(self):
        a = fixture_episode(self.definition, self.sim, self.nav)
        b = copy.deepcopy(a)
        self.assertTrue(compare_episodes(a, b)['repeatable'])
        b['observations'][0, 0, 0, 0] += 1
        result = compare_episodes(a, b)
        self.assertFalse(result['repeatable'])
        self.assertFalse(result['fields']['observations']['exact'])
        b['metadata']['seed'] += 1
        with self.assertRaises(ValueError):
            compare_episodes(a, b)

    def test_video_roundtrip_and_overwrite(self):
        episode = fixture_episode(self.definition, self.sim, self.nav)
        video = export_video(episode, self.root / 'unit-fixture.mp4', composite=True)
        import imageio.v2 as imageio
        with imageio.get_reader(str(video)) as reader:
            self.assertEqual(reader.count_frames(), 3)
            self.assertEqual(reader.get_data(0).shape, (4, 8, 3))
        with self.assertRaises(FileExistsError):
            export_video(episode, video, composite=True)

    def test_cli_keeps_evaluation_split_separate(self):
        self.assertEqual(main(['run', '--episode-data', str(self.root / 'val/val.json.gz'),
                               '--scene-data-dir', str(self.scenes), '--scenes', 'A',
                               '--output-dir', str(self.root / 'eval')]), 2)

    @unittest.skipUnless(importlib.util.find_spec('habitat') and importlib.util.find_spec('habitat_sim')
                         and os.environ.get('GIBSON_PILOT_SCENE') and os.environ.get('HABITAT_GIBSON_EPISODES')
                         and os.environ.get('HABITAT_GIBSON_SCENE_DIR'),
                         'Requires pinned Linux Habitat runtime and explicit development Gibson assets')
    def test_real_lab_episode_and_repeat(self):
        from jepa_navigation.data.gibson_pilot import run_pilot_episode
        definitions = load_pointnav(os.environ['HABITAT_GIBSON_EPISODES'],
                                    os.environ['HABITAT_GIBSON_SCENE_DIR'], 1,
                                    scenes=[os.environ['GIBSON_PILOT_SCENE']])
        first = run_pilot_episode(definitions[0], SimulatorConfig(), self.nav)
        second = run_pilot_episode(definitions[0], SimulatorConfig(), self.nav)
        from jepa_navigation.data.gibson_pilot import sha256_file
        for e in (first, second):
            e['metadata']['scene_sha256'] = sha256_file(definitions[0]['scene'])
        self.assertTrue(first['metrics']['success'], first['metrics'])
        self.assertTrue(compare_episodes(first, second)['repeatable'])


if __name__ == '__main__':
    unittest.main()

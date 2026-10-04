"""Dataset bookkeeping tests use explicit doubles, never production fallback data."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from test_habitat_smoke import FakeEnv
from data.habitat_collector import collect_episode
from data.habitat_dataset_collection import collect_dataset, episode_seed, summary, validate_dataset
from datasets.habitat_dataset import HabitatTrajectoryDataset, validate_trajectory


class SeededEnv(FakeEnv):
    def __init__(self):
        self.sim_seeds, self.pathfinder_seeds = [], []
        self.sim = SimpleNamespace(seed=self.sim_seeds.append,
                                   pathfinder=SimpleNamespace(seed=self.pathfinder_seeds.append))

    def state(self):
        state = super().state()
        state['position'][0] = self.seed
        state['heading'] = self.seed / 100.
        return state

    def expert_action(self):
        # Odd seeds deliberately time out; failed real rollouts use this same schema.
        return super().expert_action() if self.seed % 2 == 0 else 0


class DatasetTests(unittest.TestCase):
    def collect(self, root, count=3):
        env = SeededEnv()
        manifest = collect_dataset(env, root, 'scene.glb', count, 42, 2)
        return env, manifest

    def test_seeds_repeatable_varying_and_bounded(self):
        seeds = [episode_seed(42, i) for i in range(100)]
        self.assertEqual(seeds, [episode_seed(42, i) for i in range(100)])
        self.assertEqual(len(set(seeds)), 100)
        self.assertEqual(seeds[0], 42)
        self.assertEqual(episode_seed(2**31-1, 1), 0)
        with self.assertRaises(ValueError):
            episode_seed(-1, 0)

    def test_manifest_summary_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'a'
            env, manifest = self.collect(root)
            self.assertEqual(env.sim_seeds, [42, 43, 44])
            self.assertEqual(env.pathfinder_seeds, env.sim_seeds)
            self.assertEqual(validate_dataset(root), manifest)
            _, again = self.collect(Path(tmp) / 'b')
            self.assertEqual(manifest, again)
            self.assertEqual([e['trajectory'] for e in manifest['episodes']],
                             [f'episode_{i:06d}.pt' for i in range(3)])
            stats = summary(manifest['episodes'])
            self.assertEqual(stats['successes'], 2)
            self.assertEqual(stats['failures'], 1)
            self.assertAlmostEqual(stats['success_rate'], 2/3)
            self.assertEqual(stats['mean_episode_length'], 2)
            self.assertEqual(stats['median_episode_length'], 2)
            self.assertEqual(stats['mean_initial_goal_distance'], 1.)
            self.assertAlmostEqual(stats['mean_collisions'], 4/3)
            for i in range(3):
                a = torch.load(root / f'episode_{i:06d}.pt', weights_only=True)
                b = torch.load(Path(tmp) / 'b' / f'episode_{i:06d}.pt', weights_only=True)
                for key in ('positions', 'headings', 'actions', 'observations'):
                    self.assertTrue(torch.equal(a[key], b[key]))
            with self.assertRaises(FileExistsError):
                self.collect(root)

    def test_success_filter_and_window_indexing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            self.collect(root)
            ds = HabitatTrajectoryDataset(root)
            self.assertEqual(len(ds), 4)
            self.assertEqual(ds[2]['positions'][0, 0], 44.)
            self.assertEqual(len(HabitatTrajectoryDataset(root, success_only=False)), 6)
            self.assertEqual(len(HabitatTrajectoryDataset(root, num_steps=2)), 2)
            self.assertEqual(len(HabitatTrajectoryDataset(root, num_steps=3)), 0)
            self.assertEqual(ds[0]['next_positions'][0, 2], -1.)

    def test_corruption_rejected(self):
        base = collect_episode(FakeEnv())
        mutations = [lambda t: t['actions'].fill_(4),
                     lambda t: t['positions'].fill_(float('nan')),
                     lambda t: t['headings'].fill_(float('inf')),
                     lambda t: t.update(observations=t['observations'].float()),
                     lambda t: t.update(observations=t['observations'][:, :2]),
                     lambda t: t['observations'].zero_(),
                     lambda t: t.update(relative_goals=t['relative_goals'][:-1]),
                     lambda t: t['metrics'].update(steps=100),
                     lambda t: t['actions'].fill_(3)]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                episode = copy.deepcopy(base)
                mutation(episode)
                with self.assertRaises(ValueError):
                    validate_trajectory(episode)
        empty = copy.deepcopy(base)
        empty['actions'] = empty['actions'][:0]
        empty['collisions'] = empty['collisions'][:0]
        for key in ('observations', 'positions', 'headings', 'relative_goals'):
            empty[key] = empty[key][:1]
        empty['metrics'].update(steps=0)
        with self.assertRaises(ValueError):
            validate_trajectory(empty)

    def test_all_failures_and_nonfinite_final_distance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            env = SeededEnv()
            env.distance = lambda: float('inf')
            manifest = collect_dataset(env, root, 'scene.glb', 1, 43, 1)
            self.assertIsNone(manifest['episodes'][0]['final_distance'])
            self.assertEqual(validate_dataset(root), manifest)
            self.assertEqual(len(HabitatTrajectoryDataset(root)), 0)
            self.assertEqual(len(HabitatTrajectoryDataset(root, success_only=False)), 1)

    def test_stale_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            _, manifest = self.collect(root)
            manifest['episodes'][0]['success'] = False
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'manifest does not match'):
                validate_dataset(root)

    def test_duplicate_stops_with_partial_manifest(self):
        env = SeededEnv()
        # Deliberately break state variation to exercise duplicate detection.
        env.state = lambda: FakeEnv.state(env)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'
            with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
                collect_dataset(env, root, 'scene.glb', 3, 42, 2)
            manifest = validate_dataset(root)
            self.assertFalse(manifest['complete'])
            self.assertEqual(len(manifest['episodes']), 1)


if __name__ == '__main__':
    unittest.main()

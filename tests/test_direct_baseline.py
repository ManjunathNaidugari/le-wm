"""Synthetic wiring tests only; no pretrained V-JEPA/navigation result is claimed."""
from dataclasses import asdict, replace
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from jepa_navigation.baseline.audit import audit_run, combine_audits
from jepa_navigation.baseline.cache import extract_cache, validate_cache
from jepa_navigation.baseline.common import fingerprint, sha256_file, write_json
from jepa_navigation.baseline.execution import OnlinePolicy, execute_plan, learned_runner
from jepa_navigation.baseline.features import FeatureConfig, FrozenEncoder, CausalHistory, causal_indices, raw_clip
from jepa_navigation.baseline.policy import DirectPolicy
from jepa_navigation.baseline.splits import integration_split, inspect_availability, prepare_splits, validate_splits
from jepa_navigation.baseline.training import TrainConfig, train_policy, load_policy, CachedTransitions
from jepa_navigation.baseline.worker import WorkerClient, worker_command, send_message, read_message
from jepa_navigation.data.gibson_pilot import collect_pilot
from jepa_navigation.data.pointnav import load_pointnav
from jepa_navigation.navigation.actions import Action, to_lab_action
from jepa_navigation.utils.config import SimulatorConfig, NavigationConfig
from test_gibson_pilot import FixtureEnv, fixture_episode as original_fixture_episode
from test_pointnav import raw_episode, write_json
from baseline_fixtures import TestEncoder, train_fixture_policy, load_fixture_policy


def fixture_episode(*args, **kwargs):
    episode = original_fixture_episode(*args, **kwargs)
    episode['metadata'].update(controller='ShortestPathFollower(stop_on_error=False)',
                               workflow='habitat_lab_pointnav_reference',
                               habitat_sim_version='0.3.3', habitat_lab_version='0.3.3')
    # Schema-shaped test inputs live only in TemporaryDirectory; no research data is produced.
    return episode


class TinyTokenModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))
        self.dropout = nn.Dropout(0.9)
        self.config = config

    def forward(self, x):
        return self.dropout(x.mean(dim=(2, 3, 4)))[:, None, :].expand(-1, self.config.history_length // 2 * 256, -1) * self.weight


def fixture_transform(frames):
    values = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255
    return [torch.nn.functional.interpolate(values.permute(1, 0, 2, 3), size=(256, 256), mode='bilinear').permute(1, 0, 2, 3)]


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scenes = self.root / 'scenes'
        (self.scenes / 'gibson').mkdir(parents=True)
        for name in ('A', 'B', 'C', 'Adrian'):
            (self.scenes / 'gibson' / (name + '.glb')).write_bytes(b'parser fixture only')
        self.index = self.root / 'train/train.json.gz'
        write_json(self.index, dict(episodes=[], content_scenes_path='{data_path}/content/{scene}.json.gz'))
        for name in ('A', 'B', 'C', 'Adrian'):
            write_json(self.index.parent / 'content' / (name + '.json.gz'),
                       dict(episodes=[raw_episode('gibson/' + name + '.glb', str(i)) for i in range(3)]))
        self.definition = load_pointnav(self.index, self.scenes, 1, scenes=['A'], episode_ids=['0'])[0]
        self.sim, self.nav = SimulatorConfig(resolution=4), NavigationConfig()
        self.config = FeatureConfig(history_length=4, grid_size=2)
        self.encoder = TestEncoder(self.config)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.old_threads)

    def make_audit(self, name='pilot'):
        directory = self.root / name
        with contextlib.redirect_stdout(io.StringIO()):
            collect_pilot([self.definition], directory, self.sim, self.nav, runner=fixture_episode)
        report_path = self.root / (name + '-audit.json')
        report = audit_run(directory, report_path)
        return directory, report_path, report

    def make_cache(self):
        directory, audit, _ = self.make_audit()
        root = self.root / 'cache'
        extract_cache(audit, root, self.encoder, chunk_size=1)
        return directory, audit, root

    def test_causal_padding_and_history_online_match(self):
        for stride in (1, 2, 3):
            config = replace(self.config, frame_stride=stride)
            history = CausalHistory(config)
            observations = torch.stack([torch.full((3, 4, 4), i, dtype=torch.uint8) for i in range(17)])
            for t in range(17):
                online = history.append(observations[t].permute(1, 2, 0).numpy())
                np.testing.assert_array_equal(online, raw_clip(observations, t, config))
                self.assertLessEqual(causal_indices(t, config).max(), t)
            fresh = CausalHistory(config)
            self.assertTrue((fresh.append(np.full((4, 4, 3), 99, dtype=np.uint8)) == 99).all())

    def test_future_changes_cannot_change_action_t_features(self):
        observations = torch.stack([torch.full((3, 4, 4), i, dtype=torch.uint8) for i in range(8)])
        before = self.encoder.encode(raw_clip(observations, 3, self.config))
        observations[4:] = 255
        np.testing.assert_array_equal(before, self.encoder.encode(raw_clip(observations, 3, self.config)))

    def test_frozen_encoder_eval_and_no_parameter_updates(self):
        model = TinyTokenModel(self.config)
        frozen = FrozenEncoder(model, fixture_transform, self.config, self.encoder.identity)
        weights = model.weight.clone()
        clip = np.arange(4 * 3 * 4 * 4, dtype=np.uint8).reshape(4, 3, 4, 4)
        first, second = frozen.encode(clip), frozen.encode(clip)
        np.testing.assert_array_equal(first, second)
        history = CausalHistory(self.config)
        for frame in clip:
            online = history.append(frame.transpose(1, 2, 0))
        np.testing.assert_array_equal(first, frozen.encode(online))
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertIsNone(model.weight.grad)
        self.assertTrue(torch.equal(weights, model.weight))
        model.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, 'frozen'):
            frozen.encode(clip)

    def test_audit_flags_keep_valid_close_wall_or_static_examples(self):
        _, _, report = self.make_audit()
        row = report['episodes'][0]
        self.assertTrue(row['included'])
        self.assertEqual(row['action_counts'], dict(FORWARD=1, TURN_LEFT=0, TURN_RIGHT=0, STOP=1))
        self.assertEqual(row['movement_actions'], 1)
        self.assertEqual(row['collision_rate_per_movement_action'], 0.)
        self.assertEqual(row['frozen_motion_action_indices'], [0])
        self.assertEqual(row['negligible_forward_count'], 0)

    def test_audit_missing_and_exclusions_are_explicit(self):
        directory, _, _ = self.make_audit()
        report = audit_run(directory, self.root / 'excluded.json', {'A/0': 'review requested by researcher'})
        self.assertFalse(report['episodes'][0]['included'])
        self.assertIn('explicit exclusion', report['episodes'][0]['inclusion_reason'])
        (directory / 'episode_000000.pt').unlink()
        report = audit_run(directory, self.root / 'missing.json')
        self.assertTrue(report['episodes'][0]['structural_errors'])
        self.assertFalse(report['episodes'][0]['included'])

    def test_cache_alignment_raw_rgb_and_no_terminal_future(self):
        directory, audit, root = self.make_cache()
        manifest = validate_cache(root, self.encoder.identity)
        item = manifest['episodes'][0]
        episode = torch.load(directory / 'episode_000000.pt', weights_only=True)
        values = np.load(root / item['chunks'][0]['file'])
        np.testing.assert_array_equal(values[0], self.encoder.encode(raw_clip(episode['observations'], 0, self.config)))
        with np.load(root / item['targets']) as targets:
            np.testing.assert_array_equal(targets['actions'], episode['actions'])
            np.testing.assert_array_equal(targets['goals'], episode['relative_goals'][:-1])
        self.assertEqual(sum(c['end'] - c['start'] for c in item['chunks']), len(episode['actions']))

    def test_cache_resume_only_missing_chunks(self):
        directory, audit, _ = self.make_audit()
        root = self.root / 'resume'
        original = self.encoder.encode
        calls = []
        def interrupt(clip):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('simulated interruption')
            return original(clip)
        with patch.object(self.encoder, 'encode', side_effect=interrupt), self.assertRaises(RuntimeError):
            extract_cache(audit, root, self.encoder, chunk_size=1)
        with patch.object(self.encoder, 'encode', wraps=original) as spy:
            extract_cache(audit, root, self.encoder, chunk_size=1)
            self.assertEqual(spy.call_count, 1)
        validate_cache(root)

    def test_stale_source_and_encoder_cache_rejected(self):
        directory, audit, root = self.make_cache()
        incompatible = copy.deepcopy(self.encoder.identity)
        incompatible['feature_config']['frame_stride'] = 2
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            validate_cache(root, incompatible)
        with (directory / 'episode_000000.pt').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'stale'):
            validate_cache(root)
        with self.assertRaises(ValueError):
            extract_cache(audit, root, TestEncoder(replace(self.config, frame_stride=2)), chunk_size=1)

    def test_cache_corruption_and_alignment_rejected(self):
        _, _, root = self.make_cache()
        manifest = read_json_for_test(root / 'cache.json')
        chunk = manifest['episodes'][0]['chunks'][0]
        with (root / chunk['file']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Corrupt'):
            validate_cache(root)

    def make_plan(self):
        inventory = self.root / 'inventory.json'
        inspect_availability([self.index], self.scenes, inventory)
        output = self.root / 'splits.json'
        plan = prepare_splits(inventory, dict(train=['A'], development=['B'], final_evaluation=['C']),
                              dict(train=1, development=1, final_evaluation=1), output)
        return output, plan

    def test_inventory_and_materialized_split_use_available_intersections(self):
        path, plan = self.make_plan()
        self.assertEqual(plan['splits']['train']['buildings'], ['A'])
        validate_splits(plan)
        inventory = self.root / 'missing-inventory.json'
        (self.scenes / 'gibson/B.glb').unlink()
        inspect_availability([self.index], self.scenes, inventory)
        with self.assertRaisesRegex(ValueError, 'intersection'):
            prepare_splits(inventory, dict(train=['A'], development=['B'], final_evaluation=['C']),
                           dict(train=1, development=1, final_evaluation=1), self.root / 'bad.json')

    def test_split_building_leakage_duplicates_and_pilot_final_rejected(self):
        _, original = self.make_plan()
        for change in ('overlap', 'duplicate', 'pilot', 'known_pilot'):
            plan = copy.deepcopy(original)
            if change == 'overlap':
                plan['splits']['development'] = copy.deepcopy(plan['splits']['train'])
            elif change == 'duplicate':
                plan['splits']['train']['episodes'].append(plan['splits']['train']['episodes'][0])
            elif change == 'pilot':
                plan['pilot_buildings'].append('C')
            else:
                plan['pilot_buildings'] = []
                plan['splits']['final_evaluation']['buildings'] = ['Adrian']
                plan['splits']['final_evaluation']['episodes'][0]['source_scene_id'] = 'gibson/Adrian.glb'
            plan['fingerprint'] = fingerprint({k: v for k, v in plan.items() if k != 'fingerprint'})
            with self.assertRaises(ValueError):
                validate_splits(plan)

    def test_collection_resume_accounts_failures_and_preserves_attempts(self):
        plan_path, plan = self.make_plan()
        root = self.root / 'collection'
        definition = plan['splits']['train']['episodes'][0]
        def fail(d, sim, nav):
            raise RuntimeError('fixture renderer error')
        with contextlib.redirect_stdout(io.StringIO()):
            result = execute_plan(plan_path, 'train', root, self.sim, self.nav, runner=fail)
            self.assertEqual(result['summary']['runtime_errors'], 1)
            result = execute_plan(plan_path, 'train', root, self.sim, self.nav, runner=fixture_episode,
                                  resume=True, retry_errors=True)
        self.assertEqual(len(result['episodes'][0]['attempts']), 2)
        self.assertEqual(result['episodes'][0]['attempts'][0]['status'], 'error')
        self.assertEqual(result['episodes'][0]['status'], 'finished')
        with patch('jepa_navigation.baseline.execution.run_pilot_episode') as spy:
            execute_plan(plan_path, 'train', root, self.sim, self.nav, resume=True)
            spy.assert_not_called()
        with self.assertRaises(ValueError):
            execute_plan(plan_path, 'train', root, self.sim, replace(self.nav, turn_degrees=30), resume=True)
        truncated = copy.deepcopy(result)
        truncated['episodes'] = []
        (root / 'manifest.json').write_text(json.dumps(truncated))
        with self.assertRaisesRegex(ValueError, 'every requested'):
            execute_plan(plan_path, 'train', root, self.sim, self.nav, resume=True)

    def test_video_error_keeps_simulation_metrics_and_attempt_accounting(self):
        path, _ = self.make_plan()
        with patch('jepa_navigation.baseline.execution.export_video', side_effect=RuntimeError('fixture encoder failure')):
            with contextlib.redirect_stdout(io.StringIO()):
                result = execute_plan(path, 'train', self.root / 'video-error', self.sim, self.nav,
                                      runner=fixture_episode, videos=True)
        self.assertEqual(result['summary']['executed'], 1)
        self.assertEqual(result['summary']['successes'], 1)
        self.assertEqual(result['summary']['navigation_failures'], 0)
        self.assertEqual(result['summary']['video_errors'], 1)
        self.assertEqual(result['summary']['runtime_errors'], 0)
        self.assertTrue((self.root / 'video-error' / result['episodes'][0]['trajectory']).is_file())

    def test_binary_protocol_and_different_python_worker(self):
        stream = io.BytesIO()
        clip = np.arange(4 * 3 * 4 * 4, dtype=np.uint8).reshape(4, 3, 4, 4)
        send_message(stream, {'op': 'encode'}, clip)
        stream.seek(0)
        header, array = read_message(stream)
        np.testing.assert_array_equal(array, clip)
        python = os.environ.get('BASELINE_TEST_ENCODER_PYTHON', sys.executable)
        with WorkerClient([python, str(Path(__file__).with_name('feature_worker_fixture.py')),
                          '--feature-config', json.dumps(asdict(self.config)),
                          '--source-dir', 'test-only', '--checkpoint', 'test-only']) as worker:
            self.assertTrue(worker.identity['test_encoder'])
            np.testing.assert_array_equal(worker.encode(clip), self.encoder.encode(clip))

    def test_training_tiny_overfit_and_checkpoint_roundtrip(self):
        _, audit, cache = self.make_cache()
        split = self.root / 'integration.json'
        integration_split(audit, split)
        output = self.root / 'policy-training'
        config = TrainConfig(epochs=45, learning_rate=0.02, batch_size=2, hidden_dim=16,
                             projection_dim=4, weight_decay=0, tiny_steps=2, class_weights=False)
        with contextlib.redirect_stdout(io.StringIO()):
            metadata = train_fixture_policy(cache, split, output, config)
        self.assertTrue(metadata['integration_only'])
        self.assertGreaterEqual(metadata['epochs'][-1]['development']['accuracy'], 0.99)
        self.assertIn('resubstitution', metadata['validation_label'])
        model, saved = load_fixture_policy(output / 'best.pt')
        self.assertTrue(torch.allclose(model.goal_mean, torch.tensor([0.5, 0., 0.])))
        self.assertEqual(saved['actions'], ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP'])
        with self.assertRaisesRegex(ValueError, 'Test encoder'):
            train_policy(cache, split, self.root / 'unlabeled', config)

    def test_combined_audit_duplicate_rejection_and_training_only_statistics(self):
        from jepa_navigation.data.habitat_collector import collect_episode
        class DistantEnv(FixtureEnv):
            goal = np.array([0., 0., -100.], dtype=np.float32)
            initial_path = np.array([[0., 0., 0.], [0., 0., -100.]], dtype=np.float32)
            initial_distance = 100.
            def state(self):
                state = super().state()
                state['position'][2] *= 100
                state['relative_goal'][0] *= 100
                return state
            def distance(self):
                return 0. if self.moved else 100.
        definitions = [raw_episode('gibson/B.glb', str(i)) for i in range(3)]
        for definition in definitions:
            definition['goals'][0]['position'] = [0., 0., -100.]
        write_json(self.index.parent / 'content/B.json.gz', dict(episodes=definitions))
        plan_path, plan = self.make_plan()
        audits = []
        def runner(definition, sim, nav):
            episode = collect_episode(DistantEnv(), nav.max_steps)
            episode['metadata'] = fixture_episode(definition, sim, nav)['metadata']
            return episode
        for phase in ('train', 'development'):
            directory = self.root / ('collected-' + phase)
            with contextlib.redirect_stdout(io.StringIO()):
                execute_plan(plan_path, phase, directory, self.sim, self.nav,
                             runner=runner if phase == 'development' else fixture_episode)
            audit = self.root / (phase + '-audit.json')
            report = audit_run(directory, audit)
            self.assertEqual(report['summary']['included'], 1, report)
            audits.append(audit)
        combined = self.root / 'combined.json'
        self.assertEqual(combine_audits(audits, combined)['summary']['included'], 2)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            combine_audits([audits[0], audits[0]], self.root / 'duplicate.json')
        cache = self.root / 'experiment-cache'
        extract_cache(combined, cache, self.encoder, chunk_size=1)
        with contextlib.redirect_stdout(io.StringIO()):
            report = train_fixture_policy(cache, plan_path, self.root / 'normal-training',
                                  TrainConfig(epochs=1, batch_size=2))
        with self.assertRaisesRegex(ValueError, 'Test encoder'):
            load_policy(self.root / 'normal-training/last.pt')
        model, _ = load_fixture_policy(self.root / 'normal-training/last.pt')
        self.assertTrue(torch.allclose(model.goal_mean, torch.tensor([0.5, 0., 0.])))
        self.assertEqual(report['epochs'][0]['development']['samples'], 2)
        self.assertIn('building-disjoint', report['validation_label'])

    def test_learned_runner_records_latency_and_does_not_call_expert(self):
        from jepa_navigation.data.habitat_collector import collect_episode
        model = DirectPolicy(feature_dim=3, cells=8, goal_only=True).eval()
        model.forward = lambda features, goals: torch.stack([goals[:, 0], torch.zeros(len(goals)),
                                                            torch.zeros(len(goals)), 0.5-goals[:, 0]], dim=1)
        def run(definition, sim, nav, controller):
            env = FixtureEnv()
            with patch.object(env, 'expert_action', side_effect=AssertionError('Expert access')):
                episode = collect_episode(env, nav.max_steps, controller=controller)
            episode['metadata'] = fixture_episode(definition, sim, nav)['metadata']
            return episode
        metadata = dict(encoder=self.encoder.identity, integration_only=True)
        with patch('jepa_navigation.baseline.execution.run_pilot_episode', side_effect=run):
            episode = learned_runner(model, metadata)(self.definition, self.sim, self.nav)
        self.assertEqual(episode['actions'].tolist(), [0, 3])
        self.assertEqual(episode['metrics']['termination'], 'stop')
        self.assertEqual(len(episode['inference_latency_ms']), 2)
        self.assertTrue((episode['inference_latency_ms'] >= 0).all())
        self.assertEqual(episode['metadata']['controller'], 'goal_only_policy')

    def test_tiny_selection_contains_stop_even_when_prefix_does_not(self):
        _, _, cache_dir = self.make_cache()
        cache = validate_cache(cache_dir)
        item = cache['episodes'][0]
        with np.load(cache_dir / item['targets']) as targets:
            goals = targets['goals'].copy()
        # Dataset-only fixture: exercise selection over a long forward prefix.
        np.savez(cache_dir / item['targets'], actions=np.array([0, 0, 0, 3]),
                 goals=np.concatenate([goals[:1]]*3 + [goals[1:]]))
        item['steps'] = 4
        selected = CachedTransitions(cache_dir, cache, [('a', '0')], limit=2)
        self.assertEqual(selected.goals_and_actions()[1].tolist(), [0, 3])

    def test_goal_only_ignores_visual_inputs(self):
        model = DirectPolicy(feature_dim=3, cells=8, goal_only=True).eval()
        goals = torch.ones(2, 3)
        self.assertTrue(torch.equal(model(torch.zeros(2, 8, 3), goals), model(torch.ones(2, 8, 3) * 1000, goals)))
        self.assertTrue(torch.equal(model(None, goals), model(torch.randn(2, 8, 3), goals)))

    def test_online_policy_causal_features_and_privileged_input_rejection(self):
        model = DirectPolicy(feature_dim=3, cells=8, hidden_dim=8)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.head[-1].bias[3] = 10
        policy = OnlinePolicy(model, dict(encoder=self.encoder.identity), self.encoder)
        action = policy(dict(rgb=np.zeros((4, 4, 3), dtype=np.uint8), relative_goal=np.zeros(3)))
        self.assertEqual(action, Action.STOP)
        self.assertEqual(to_lab_action(action), 0)
        self.assertEqual(len(policy.latencies_ms), 1)
        with self.assertRaises(ValueError):
            policy(dict(rgb=np.zeros((4, 4, 3), dtype=np.uint8), relative_goal=np.zeros(3), shortest_path=[]))
        with self.assertRaises(ValueError):
            OnlinePolicy(model, dict(encoder=dict(backbone='different')), self.encoder)

    def test_controller_stop_terminates_without_expert_access(self):
        from jepa_navigation.data.habitat_collector import collect_episode
        env = FixtureEnv()
        with patch.object(env, 'expert_action', side_effect=AssertionError('privileged expert must never be called')):
            episode = collect_episode(env, controller=lambda state: Action.STOP)
        self.assertEqual(episode['actions'].tolist(), [3])
        self.assertEqual(episode['metrics']['termination'], 'stop')
        self.assertEqual(len(episode['observations']), 2)
        self.assertTrue(torch.equal(episode['observations'][0], episode['observations'][1]))


def read_json_for_test(path):
    return json.loads(Path(path).read_text())


if __name__ == '__main__':
    unittest.main()

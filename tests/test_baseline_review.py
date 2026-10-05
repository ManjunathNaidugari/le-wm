"""Regressions for expert eligibility, empty labels and checkpoint selection."""
import contextlib
import copy
import io
from unittest import TestCase
from unittest.mock import patch
import torch
import test_direct_baseline as fixtures
from jepa_navigation.baseline.audit import audit_run
from jepa_navigation.baseline.cache import extract_cache
from jepa_navigation.baseline.cli import main
from jepa_navigation.baseline.common import fingerprint, read_json, sha256_file, write_json
from jepa_navigation.baseline.provenance import preflight_audit
from jepa_navigation.baseline.splits import integration_split, resolve_training_inputs, training_definitions
from jepa_navigation.baseline.training import TrainConfig, train_policy
from baseline_fixtures import train_fixture_policy


class ReviewTests(TestCase):
    setUp = fixtures.BaselineTests.setUp
    make_audit = fixtures.BaselineTests.make_audit
    make_cache = fixtures.BaselineTests.make_cache
    make_plan = fixtures.BaselineTests.make_plan

    def test_synthetic_recording_cannot_be_resolved_as_expert(self):
        directory, _, _ = self.make_audit()
        self.change_recording(directory, lambda e: e['metadata'].update(synthetic=True))
        resolution = {'A/0': dict(kind='expert', reason='test override', evidence='test only')}
        row = audit_run(directory, self.root/'synthetic.json', expert_resolutions=resolution)['episodes'][0]
        self.assertFalse(row['included'])
        self.assertIn('synthetic', row['inclusion_reason'])

    def test_production_training_rejects_test_features(self):
        _, audit, cache = self.make_cache()
        split = self.root/'split.json'
        integration_split(audit, split)
        with self.assertRaisesRegex(ValueError, 'Test encoder'):
            train_policy(cache, split, self.root/'forbidden', TrainConfig(epochs=1, tiny_steps=2))
        self.assertFalse((self.root/'forbidden').exists())

    def test_encoder_check_uses_recorded_rgb(self):
        directory, _, _ = self.make_audit()
        episode = torch.load(directory/'episode_000000.pt', weights_only=True)
        with patch('jepa_navigation.baseline.cli.make_worker') as factory, contextlib.redirect_stdout(io.StringIO()):
            worker = factory.return_value.__enter__.return_value
            worker.identity = self.encoder.identity
            worker.encode.side_effect = self.encoder.encode
            code = main(['check-encoder', '--trajectory', str(directory/'episode_000000.pt'),
                         '--encoder-python', 'test-only-worker'])
        self.assertEqual(code, 0)
        clip = worker.encode.call_args.args[0]
        self.assertTrue((clip == episode['observations'][0].numpy()).all())

    def change_recording(self, directory, mutate, workflow=None, signature=None):
        path = directory/'episode_000000.pt'
        episode = torch.load(path, weights_only=True)
        mutate(episode)
        torch.save(episode, path)
        manifest = read_json(directory/'manifest.json')
        manifest['episodes'][0].update(steps=len(episode['actions']), success=episode['metrics']['success'])
        if workflow:
            manifest['workflow'] = workflow
        if signature:
            manifest['signature'] = signature
        write_json(directory/'manifest.json', manifest)

    def test_genuine_legacy_pilot_metadata_is_compatible(self):
        directory, _, _ = self.make_audit()
        def genuine(episode):
            episode['metadata'].pop('synthetic', None)
            episode['metadata'].update(habitat_sim_version='0.3.3', habitat_lab_version='0.3.3')
        self.change_recording(directory, genuine)
        output = self.root/'genuine.json'
        row = audit_run(directory, output)['episodes'][0]
        self.assertTrue(row['recording_valid'])
        self.assertTrue(row['expert_eligible'])
        self.assertTrue(row['included'])
        self.assertIsNone(row['provenance']['resolution'])
        self.assertEqual(len(preflight_audit(output)), 1)

    def test_learned_recording_is_valid_but_never_an_expert_label(self):
        directory, _, _ = self.make_audit()
        self.change_recording(directory, lambda e:e['metadata'].update(controller='goal_only_policy'),
                              'navigation_collection', dict(controller='checkpoint-digest', purpose='integration_rollout', phase='train'))
        audit = self.root/'learned.json'
        row = audit_run(directory, audit)['episodes'][0]
        self.assertTrue(row['recording_valid'])
        self.assertFalse(row['expert_eligible'])
        self.assertFalse(row['included'])
        with patch.object(self.encoder, 'encode') as spy, self.assertRaises(ValueError):
            extract_cache(audit, self.root/'forbidden-cache', self.encoder)
        spy.assert_not_called()
        resolution = {'A/0':dict(kind='expert', reason='incorrect override', evidence='fixture')}
        self.assertFalse(audit_run(directory, self.root/'no-override.json', expert_resolutions=resolution)['episodes'][0]['expert_eligible'])

    def test_unknown_provenance_requires_documented_resolution(self):
        directory, _, _ = self.make_audit()
        self.change_recording(directory, lambda e:e['metadata'].update(controller='unregistered expert'))
        self.assertFalse(audit_run(directory, self.root/'unknown.json')['episodes'][0]['included'])
        resolutions = {'A/0':dict(kind='expert', reason='inspected original log', evidence='fixture-log:12')}
        audit = self.root/'resolved.json'
        row = audit_run(directory, audit, expert_resolutions=resolutions)['episodes'][0]
        self.assertTrue(row['included'])
        cache = extract_cache(audit, self.root/'resolved-cache', self.encoder)
        self.assertEqual(cache['episodes'][0]['provenance'], row['provenance'])

    def test_zero_transitions_fail_before_worker_startup(self):
        directory, _, _ = self.make_audit()
        def zero(e):
            for key in ('observations','positions','headings','relative_goals','rotations_xyzw','visualization_frames'):
                e[key] = e[key][:1]
            e['actions'], e['collisions'] = e['actions'][:0], e['collisions'][:0]
            e['metrics'].update(success=False, steps=0, termination='follower_error', error='fixture follower failure', path_length=0., final_distance=1.)
            e['metrics']['habitat_lab'].update(success=0., spl=0., distance_to_goal=1.)
        self.change_recording(directory, zero)
        audit = self.root/'zero.json'
        row = audit_run(directory, audit)['episodes'][0]
        self.assertTrue(row['recording_valid'])
        self.assertFalse(row['training_eligible'])
        self.assertIn('zero-transition', row['inclusion_reason'])
        with patch('jepa_navigation.baseline.cli.make_worker') as worker:
            code = main(['extract','--audit',str(audit),'--cache-dir',str(self.root/'zero-cache'),
                         '--encoder-python','not-an-executable'])
        self.assertEqual(code, 2)
        worker.assert_not_called()
        with self.assertRaises(ValueError):
            integration_split(audit, self.root/'shrunk.json')

    def test_all_selected_inputs_preflight_before_any_encoding(self):
        _, audit, report = self.make_audit()
        bad = copy.deepcopy(report['episodes'][0])
        bad.update(identity='A/999', source_episode_id='999', trajectory=str(self.root/'missing.pt'))
        report['episodes'].append(bad)
        write_json(audit, report)
        with patch.object(self.encoder,'encode') as encoder, self.assertRaisesRegex(ValueError,'A/999'):
            extract_cache(audit, self.root/'partial-cache', self.encoder)
        encoder.assert_not_called()
        self.assertFalse((self.root/'partial-cache').exists())

    def test_training_rechecks_raw_controller_and_cache_provenance(self):
        directory, audit, cache_root = self.make_cache()
        split = self.root/'integration.json'
        integration_split(audit, split)
        path = directory/'episode_000000.pt'
        episode = torch.load(path, weights_only=True)
        episode['metadata']['controller'] = 'goal_only_policy'
        torch.save(episode, path)
        cache = read_json(cache_root/'cache.json')
        cache['episodes'][0]['source_sha256'] = sha256_file(path)
        cache['signature']['sources'][0]['sha256'] = sha256_file(path)
        cache['fingerprint'] = fingerprint(cache['signature'])
        write_json(cache_root/'cache.json', cache)
        with self.assertRaisesRegex(ValueError,'Expert provenance'):
            train_policy(cache_root, split, self.root/'bad-training', TrainConfig(epochs=1,tiny_steps=2))
        self.assertFalse((self.root/'bad-training').exists())

    def test_checkpoint_own_epoch_and_best_epoch_are_consistent(self):
        _, audit, cache = self.make_cache()
        split = self.root/'integration.json'
        integration_split(audit, split)
        metrics = dict(loss=3., accuracy=.5, samples=2, confusion_matrix=[[1,0,0,0],[0,0,0,0],[0,0,0,0],[1,0,0,0]])
        outputs = [dict(metrics,loss=loss) for loss in (3.,3.,2.,1.,1.,2.)]
        root = self.root/'selection'
        with patch('jepa_navigation.baseline.training.run_epoch', side_effect=outputs), contextlib.redirect_stdout(io.StringIO()):
            train_fixture_policy(cache, split, root, TrainConfig(epochs=3,tiny_steps=2))
        final = read_json(root/'training.json')
        best, last = [torch.load(root/(name+'.pt'),weights_only=True) for name in ('best','last')]
        self.assertEqual((final['checkpoint_epoch'], final['best_epoch']), (3,2))
        self.assertEqual((best['epoch'],best['metadata']['checkpoint_epoch'],best['metadata']['best_epoch']), (2,2,2))
        self.assertEqual((last['epoch'],last['metadata']['checkpoint_epoch'],last['metadata']['best_epoch']), (3,3,2))
        self.assertNotIn('experiment_record', final)
        self.assertNotIn('measurement', final['epochs'][0]['train'])

    def test_label_exclusions_preserve_all_evaluation_requests(self):
        split, original = self.make_plan()
        rows = []
        for phase in ('train','development'):
            definition = original['splits'][phase]['episodes'][0]
            rows.append(dict(building=fixtures.Path(definition['source_scene_id']).stem,
                             source_episode_id=definition['source_episode_id'], included=phase=='train',
                             inclusion_reason='zero-transition recording'))
        audit = self.root/'eligibility.json'
        write_json(audit, dict(schema_version=2, workflow='baseline_audit', episodes=rows))
        revised = resolve_training_inputs(split, audit, self.root/'resolved-splits.json', 'reviewed label eligibility')
        self.assertEqual(revised['splits'], original['splits'])
        self.assertEqual(training_definitions(revised,'development'), [])
        self.assertNotEqual(revised['fingerprint'], original['fingerprint'])

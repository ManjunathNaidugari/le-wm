import tempfile
from pathlib import Path
import unittest
import torch

from data.rgb_qc import rgb_quality, qc_dataset
from data.habitat_collector import collect_episode
from test_habitat_smoke import FakeEnv


class RGBQualityTests(unittest.TestCase):
    def trajectory(self):
        t = collect_episode(FakeEnv())
        gen = torch.Generator().manual_seed(9)
        t['observations'] = torch.randint(20, 230, (6, 3, 8, 8), dtype=torch.uint8, generator=gen)
        t['positions'] = torch.tensor([[0., y, 0.] for y in (0., .1, .2, .1, 0., 0.)])
        t['actions'] = torch.tensor([0, 1, 0, 2, 3])
        return t

    def test_normal_metrics_dimensions_and_y(self):
        t = self.trajectory()
        r = rgb_quality(t)
        self.assertEqual(r['qc_status'], 'pass')
        self.assertEqual(r['image_shape'], [6, 3, 8, 8])
        self.assertEqual(r['black_ratio'], 0.)
        self.assertEqual(r['saturation_ratio'], 0.)
        self.assertAlmostEqual(r['mean_brightness'], t['observations'].float().mean().item(), places=4)
        self.assertAlmostEqual(r['frames'][0]['variance'], t['observations'][0].float().var(unbiased=False).item(), places=3)
        self.assertAlmostEqual(r['max_y'], .2)
        self.assertEqual(r['min_y'], 0.)

    def test_black_and_saturated_frames(self):
        for value, flag in [(0, 'near_black_frames'), (255, 'saturated_frames')]:
            t = self.trajectory()
            t['observations'][2].fill_(value)
            r = rgb_quality(t)
            self.assertIn(flag, r['flags'])
            self.assertEqual(r[flag], 1)
            self.assertEqual(r['qc_status'], 'flagged')

    def test_frozen_and_stop_exclusion(self):
        t = self.trajectory()
        t['observations'][:] = t['observations'][0].clone()
        r = rgb_quality(t)
        self.assertIn('frozen_motion_frames', r['flags'])
        self.assertEqual(r['motion_pairs'], 4)
        self.assertEqual(r['frozen_ratio'], 1.)
        t = self.trajectory()
        t['observations'][-1] = t['observations'][-2]
        r = rgb_quality(t)
        self.assertAlmostEqual(r['frozen_ratio'], .2)
        self.assertEqual(r['motion_frozen_ratio'], 0.)
        self.assertNotIn('frozen_motion_frames', r['flags'])

    def test_nan_and_invalid_dimensions(self):
        t = self.trajectory()
        t['observations'] = t['observations'].float()
        t['observations'][0, 0, 0, 0] = float('nan')
        r = rgb_quality(t)
        self.assertEqual(r['nonfinite_values'], 1)
        self.assertIn('nonfinite_rgb', r['flags'])
        t['observations'] = torch.zeros(3, 8, 8)
        self.assertIn('invalid_rgb_dimensions', rgb_quality(t)['flags'])

    def test_report_keeps_black_success_and_unreadable_file(self):
        t = collect_episode(FakeEnv())
        t['observations'].zero_()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(t, root / 'episode_000000.pt')
            (root / 'episode_000001.pt').write_text('broken')
            report = qc_dataset(root)
            self.assertEqual(report['flagged'], 2)
            self.assertIn('near_black_frames', report['episodes'][0]['flags'])
            self.assertEqual(report['episodes'][1]['flags'], ['unreadable_trajectory'])
            self.assertTrue((root / 'qc_report.json').is_file())
            self.assertEqual(len(list(root.glob('*.pt'))), 2)

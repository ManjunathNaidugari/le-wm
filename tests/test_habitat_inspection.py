"""Exercise the actual inspection CLI and selective MP4 export on unit fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import imageio.v2 as imageio
import torch

from jepa_navigation.data.habitat_dataset_collection import collect_dataset
from jepa_navigation.utils.inspection_cli import main
from test_habitat_dataset import SeededEnv


class InspectionTests(unittest.TestCase):
    def test_qc_and_selected_exports_preserve_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp) / 'episodes'
            collect_dataset(SeededEnv(), root, 'unit-test-only.glb', 2, 42, 2)
            # Structurally valid black success should be flagged, never discarded.
            path = root / 'episode_000000.pt'
            episode = torch.load(path, weights_only=True)
            episode['observations'].zero_()
            torch.save(episode, path)
            original = path.read_bytes()
            with patch('sys.argv', ['inspect', '--dataset-dir', str(root), '--qc', '--export-flagged']):
                self.assertEqual(main(), 0)
            report = json.loads((root / 'qc_report.json').read_text())
            self.assertEqual(report['flagged'], 2)
            for index in range(2):
                video = root / 'videos' / f'episode_{index:06d}.mp4'
                with imageio.get_reader(str(video)) as reader:
                    self.assertEqual(reader.count_frames(), 3)
            self.assertEqual(original, path.read_bytes())
            # Existing videos are safely skipped, and multi-ID selection works.
            with patch('sys.argv', ['inspect', '--dataset-dir', str(root), '--episode', '0', '1']):
                self.assertEqual(main(), 0)

    def test_corrupt_file_still_gets_qc_report(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp) / 'episodes'
            collect_dataset(SeededEnv(), root, 'unit-test-only.glb', 1, 42, 2)
            path = root / 'episode_000000.pt'
            episode = torch.load(path, weights_only=True)
            episode['positions'][0, 0] = float('nan')
            torch.save(episode, path)
            with patch('sys.argv', ['inspect', '--dataset-dir', str(root), '--qc']):
                self.assertEqual(main(), 2)
            report = json.loads((root / 'qc_report.json').read_text())
            self.assertIn('invalid_trajectory', report['episodes'][0]['flags'])

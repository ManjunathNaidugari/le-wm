"""Test-only encoder and isolated mocks; never installed with the application."""
from dataclasses import asdict
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F

class TestEncoder:
    """Explicit local wiring fixture. It is not V-JEPA and carries a distinct identity."""
    def __init__(self, config):
        self.config = config
        self.identity = dict(backbone='TEST_ENCODER_NOT_VJEPA', test_encoder=True,
                             feature_config=asdict(config), feature_dimension=3)

    def encode(self, clip):
        clip = np.asarray(clip)
        if clip.shape[:2] != (self.config.history_length, 3) or clip.dtype != np.uint8:
            raise ValueError('Invalid test clip')
        spatial = torch.as_tensor(clip.copy()).float() / 255
        views = torch.stack([spatial.mean(0), spatial[-1]])
        values = F.adaptive_avg_pool2d(views, self.config.grid_size)
        return values.permute(0, 2, 3, 1).reshape(2 * self.config.grid_size**2, 3).numpy()


def train_fixture_policy(*args, **kwargs):
    from jepa_navigation.baseline.training import train_policy
    with patch("jepa_navigation.baseline.training.require_real_encoder"):
        return train_policy(*args, **kwargs)

def load_fixture_policy(*args, **kwargs):
    from jepa_navigation.baseline.training import load_policy
    with patch("jepa_navigation.baseline.training.require_real_encoder"):
        return load_policy(*args, **kwargs)

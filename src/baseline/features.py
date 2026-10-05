"""Causal raw-RGB history and frozen official V-JEPA 2 ViT-L/16 features."""
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path
import importlib.metadata
import subprocess
import contextlib
import sys
import numpy as np
import torch
import torch.nn.functional as F
from .common import sha256_file

SOURCE_REVISION = '204698b45b3712590f06245fbfba32d3be539812'
CHECKPOINT_REVISION = 'b3c1679b7c34d3255ef3547f27c7b226aefab26f'
CHECKPOINT_SHA256 = '5346856ec9df69487fe72a25bf2632aaa8112df33fb67708e3f7374edc1f7012'
CHECKPOINT_URL = f'https://huggingface.co/facebook/vjepa2-vitl-fpc64-256/resolve/{CHECKPOINT_REVISION}/original/model.pth'
FEATURE_PINS = {'torch': '2.4.1', 'torchvision': '0.19.1', 'timm': '1.0.9',
                'einops': '0.8.0', 'numpy': '1.26.4', 'Pillow': '10.4.0',
                'opencv-python-headless': '4.10.0.84'}


@dataclass(frozen=True)
class FeatureConfig:
    history_length: int = 64
    frame_stride: int = 1
    crop_size: int = 256
    grid_size: int = 4
    precision: str = 'float32'
    padding: str = 'repeat_first'
    pooling: str = 'temporal_mean_and_latest_spatial_grid'

    def __post_init__(self):
        if type(self.history_length) is not int or self.history_length < 2 or self.history_length % 2:
            raise ValueError('history_length must be a positive even number')
        if type(self.frame_stride) is not int or self.frame_stride < 1:
            raise ValueError('frame_stride must be positive')
        if self.crop_size != 256 or self.grid_size not in (1, 2, 4, 8, 16):
            raise ValueError('V-JEPA 2 ViT-L uses 256px input and a grid dividing 16')
        if self.padding != 'repeat_first' or self.pooling != 'temporal_mean_and_latest_spatial_grid':
            raise ValueError('Unsupported causal padding/pooling')
        if self.precision not in ('float32', 'bfloat16'):
            raise ValueError('precision must be float32 or bfloat16')


def causal_indices(t, config):
    if type(t) is not int or t < 0:
        raise ValueError('t must be a nonnegative action index')
    return np.maximum(0, t - np.arange(config.history_length - 1, -1, -1) * config.frame_stride)


def raw_clip(observations, t, config):
    if observations.dtype != torch.uint8 or observations.ndim != 4 or observations.shape[1] != 3:
        raise ValueError('Only raw uint8 RGB [states,3,H,W] may enter the encoder')
    if t >= len(observations):
        raise ValueError('Action index exceeds this episode')
    return observations[torch.as_tensor(causal_indices(t, config))].contiguous().numpy()


class CausalHistory:
    def __init__(self, config):
        self.config = config
        self.frames = deque(maxlen=1 + (config.history_length - 1) * config.frame_stride)
        self.steps = 0

    def append(self, rgb):
        value = np.asarray(rgb)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError('Online sensor RGB must be uint8 HWC')
        self.frames.append(value.transpose(2, 0, 1).copy())
        self.steps += 1
        frames = list(self.frames)
        indices = np.maximum(0, len(frames) - 1 - np.arange(self.config.history_length - 1, -1, -1) * self.config.frame_stride)
        return np.stack([frames[i] for i in indices])


def pool_tokens(tokens, config):
    tubes = config.history_length // 2
    if tokens.ndim != 3 or tokens.shape[1] != tubes * 16 * 16:
        raise ValueError('Expected V-JEPA 2 time-major 16x16 patch tokens')
    spatial = tokens.reshape(len(tokens), tubes, 16, 16, tokens.shape[-1])
    views = torch.stack([spatial.mean(dim=1), spatial[:, -1]], dim=1)
    views = views.permute(0, 1, 4, 2, 3).reshape(-1, tokens.shape[-1], 16, 16)
    views = F.adaptive_avg_pool2d(views, config.grid_size)
    return views.reshape(len(tokens), 2, tokens.shape[-1], config.grid_size, config.grid_size).permute(0, 1, 3, 4, 2).reshape(len(tokens), 2 * config.grid_size**2, -1)


class FrozenEncoder:
    def __init__(self, model, preprocessor, config, encoder_identity, device='cpu'):
        self.model = model.to(device).eval().requires_grad_(False)
        self.preprocessor, self.config = preprocessor, config
        self.device, self.identity = torch.device(device), encoder_identity

    def encode(self, clip):
        clip = np.asarray(clip)
        if clip.dtype != np.uint8 or clip.ndim != 4 or clip.shape[:2] != (self.config.history_length, 3):
            raise ValueError('Expected one causal raw-RGB clip [history,3,H,W]')
        self.model.eval()
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Encoder parameters must remain frozen')
        # Exactly the official deterministic eval transform, used both offline and online.
        transformed = self.preprocessor(list(clip.transpose(0, 2, 3, 1)))[0]
        if transformed.shape != (3, self.config.history_length, 256, 256):
            raise ValueError('Official preprocessor returned unexpected dimensions')
        if self.config.precision == 'bfloat16' and self.device.type != 'cuda':
            raise ValueError('bfloat16 inference is supported only in the CUDA worker')
        context = torch.autocast('cuda', dtype=torch.bfloat16) if self.config.precision == 'bfloat16' else contextlib.nullcontext()
        with torch.inference_mode(), context:
            tokens = self.model(transformed.unsqueeze(0).to(self.device))
            values = pool_tokens(tokens, self.config)[0].float().cpu().numpy().copy()
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite encoder features')
        return values


def official_encoder(source_dir, checkpoint, config, device='cuda'):
    source_dir = Path(source_dir).resolve()
    revision = subprocess.check_output(['git', '-C', str(source_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f'Expected official V-JEPA revision {SOURCE_REVISION}, found {revision}')
    subprocess.run(['git', '-C', str(source_dir), 'diff', '--exit-code', 'HEAD'], check=True, stdout=subprocess.DEVNULL)
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint SHA256 differs from the pinned official V-JEPA 2 ViT-L original checkpoint')
    versions = {name: importlib.metadata.version(name) for name in FEATURE_PINS}
    if versions != FEATURE_PINS:
        raise ValueError(f'Use the isolated feature environment pins: {versions}')
    with contextlib.redirect_stdout(sys.stderr):
        # Hub factory is the original V-JEPA 2 entry; 2.1 is never selected.
        # Explicit local weights bypass the current upstream localhost test download URL.
        encoder, unused_predictor = torch.hub.load(str(source_dir), 'vjepa2_vit_large', source='local', pretrained=False)
        del unused_predictor
        preprocessor = torch.hub.load(str(source_dir), 'vjepa2_preprocessor', source='local', crop_size=256)
    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint_data['target_encoder']
    state = {k.replace('module.', '').replace('backbone.', ''): v for k, v in state.items()}
    incompatible = encoder.load_state_dict(state, strict=False)
    if incompatible.missing_keys or set(incompatible.unexpected_keys) - {'pos_embed'}:
        raise ValueError(f'Unexpected encoder checkpoint keys: {incompatible}')
    del checkpoint_data, state
    encoder_identity = dict(backbone='official_vjepa2_vit_large', source_revision=revision,
                            checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_url=CHECKPOINT_URL,
                            checkpoint_key='target_encoder', feature_config=asdict(config),
                            preprocessing='official vjepa2_preprocessor: resize short side 292 bilinear; center crop 256; /255; ImageNet normalization',
                            packages=versions, feature_dimension=1024, test_encoder=False)
    return FrozenEncoder(encoder, preprocessor, config, encoder_identity, device)


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

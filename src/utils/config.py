"""Strict YAML configuration with CLI overrides; relative paths use the current directory."""
from dataclasses import asdict, dataclass, fields, replace
import math
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SimulatorConfig:
    resolution: int = 224
    camera_height_m: float = 1.5
    hfov_degrees: float = 90.
    agent_height_m: float = 1.5
    agent_radius_m: float = 0.1

    def __post_init__(self):
        if type(self.resolution) is not int or self.resolution < 2 or self.resolution % 2:
            raise ValueError('resolution must be a positive even integer')
        for key in ('camera_height_m', 'hfov_degrees', 'agent_height_m', 'agent_radius_m'):
            positive(getattr(self, key), key)
        if self.hfov_degrees >= 180:
            raise ValueError('hfov_degrees must be below 180')


@dataclass(frozen=True)
class NavigationConfig:
    seed: int = 42
    max_steps: int = 500
    forward_step_m: float = 0.25
    turn_degrees: float = 15.
    goal_radius_m: float = 0.2
    max_y_delta_m: float = 0.1
    fps: int = 10
    scene_data_dir: str = '/workspace/datasets/scene_datasets'
    episode_data: str = '/workspace/datasets/pointnav/gibson/v1/val/val.json.gz'
    output_dir: str = 'outputs/gibson_episode'

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError('seed must be an integer in [0, 2**31)')
        for key in ('max_steps', 'fps'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('forward_step_m', 'turn_degrees', 'goal_radius_m', 'max_y_delta_m'):
            positive(getattr(self, key), key)
        if self.turn_degrees > 180:
            raise ValueError('turn_degrees must be at most 180')
        for key in ('scene_data_dir', 'episode_data', 'output_dir'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f'{key} must be a nonempty path string')


def positive(value, key):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{key} must be finite and positive')


def load_config(path, cls, overrides=None):
    values = {}
    if path is not None:
        with Path(path).open() as stream:
            values = yaml.safe_load(stream)
        if not isinstance(values, dict):
            raise ValueError(f'{path}: expected a YAML mapping')
    valid = {f.name for f in fields(cls)}
    if set(values) - valid:
        raise ValueError(f'{path}: unknown keys {set(values) - valid}')
    values.update({k: str(v) if isinstance(v, Path) else v for k, v in (overrides or {}).items() if v is not None})
    return cls(**values)


def effective_settings(simulator, navigation):
    result = dict(simulator=asdict(simulator), navigation=asdict(navigation))
    for key in ('scene_data_dir', 'episode_data', 'output_dir'):
        result['navigation'][key] = str(Path(result['navigation'][key]).expanduser().resolve())
    return result

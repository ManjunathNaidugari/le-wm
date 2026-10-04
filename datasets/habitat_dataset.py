"""Transition windows for smoke trajectories; not a LeWM training adapter."""
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


def validate_trajectory(traj):
    if traj.get("schema_version") != 1:
        raise ValueError("Unsupported trajectory: recollect with scripts/smoke_habitat.py")
    required = {'actions', 'observations', 'positions', 'headings', 'relative_goals',
                'goal_position', 'collisions', 'shortest_path', 'metrics', 'action_names'}
    if not required.issubset(traj):
        raise ValueError(f"Missing trajectory fields: {required - traj.keys()}")
    actions = traj['actions']
    if not isinstance(actions, torch.Tensor) or actions.dtype != torch.int64 or actions.ndim != 1:
        raise ValueError('Expected int64 discrete action IDs 0..3')
    n = len(actions)
    if ((actions < 0) | (actions > 3)).any():
        raise ValueError('Expected int64 discrete action IDs 0..3')
    for key in ('observations', 'positions', 'headings', 'relative_goals'):
        if not isinstance(traj[key], torch.Tensor) or traj[key].ndim == 0 or len(traj[key]) != n + 1:
            raise ValueError(f'{key} must contain T+1 states for T actions')
    rgb = traj['observations']
    if rgb.dtype != torch.uint8 or rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape[2:]) < 1:
        raise ValueError('RGB must be non-empty uint8 [T+1, 3, H, W]')
    for key, shape in [('positions', (n+1, 3)), ('headings', (n+1,)),
                       ('relative_goals', (n+1, 3)), ('goal_position', (3,))]:
        value = traj[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f'{key}: expected finite floating-point tensor {shape}')
    path = traj['shortest_path']
    if not isinstance(path, torch.Tensor) or path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not torch.isfinite(path).all():
        raise ValueError('Invalid shortest path')
    collisions = traj['collisions']
    if not isinstance(collisions, torch.Tensor) or collisions.dtype != torch.bool or tuple(collisions.shape) != (n,):
        raise ValueError('collisions must contain T boolean entries')
    if traj['action_names'] != ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']:
        raise ValueError('Unexpected action vocabulary')
    m = traj['metrics']
    if type(m.get('success')) is not bool or m.get('steps') != n or m.get('termination') not in ('stop', 'max_steps', 'follower_error'):
        raise ValueError('Invalid episode metrics')
    for key in ('path_length', 'initial_geodesic_distance', 'final_distance'):
        value = m.get(key)
        if not isinstance(value, (float, int)) or math.isnan(value) or value < 0 or (not math.isfinite(value) and (key != 'final_distance' or m['success'])):
            raise ValueError(f'Invalid metric {key}')
    stop_indices = (actions == 3).nonzero().flatten().tolist()
    if stop_indices != ([n-1] if m['termination'] == 'stop' and n else []):
        raise ValueError('STOP must occur exactly at terminal stop')
    if m['termination'] == 'stop' and not n:
        raise ValueError('STOP requires an action')
    if m['success'] and (not n or m['termination'] != 'stop'):
        raise ValueError('Successful episodes must be non-empty and end in STOP')
    radius = traj.get('metadata', {}).get('goal_radius_m', 0.2)
    if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid goal radius')
    if m['success'] and m['final_distance'] > radius:
        raise ValueError('Success outside goal radius')
    if m['success'] and rgb.min() == rgb.max():
        raise ValueError('Successful episode RGB is entirely constant')


class HabitatTrajectoryDataset(Dataset):
    """Consecutive transition windows; exclude failed rollouts by default."""
    def __init__(self, trajectory_dir, num_steps=1, success_only=True):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        self.num_steps = num_steps
        self.traj_files = sorted(Path(trajectory_dir).glob("*.pt"))
        if not self.traj_files:
            raise ValueError(f"No trajectories in {trajectory_dir}")
        self.indices = []
        for i, path in enumerate(self.traj_files):
            traj = torch.load(path, map_location="cpu", weights_only=True)
            validate_trajectory(traj)
            if success_only and not traj['metrics']['success']:
                continue
            self.indices.extend((i, start) for start in range(len(traj["actions"]) - num_steps + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, start = self.indices[index]
        traj = torch.load(self.traj_files[i], map_location="cpu", weights_only=True)
        end = start + self.num_steps
        item = {"actions": traj["actions"][start:end],
                "collisions": traj["collisions"][start:end],
                "goal_position": traj["goal_position"]}
        for key in ("observations", "positions", "headings", "relative_goals"):
            item[key] = traj[key][start:end]
            item["next_" + key] = traj[key][start + 1:end + 1]
        return item


def create_habitat_dataloader(trajectory_dir, batch_size=1, num_steps=1, num_workers=0, shuffle=False, success_only=True):
    return DataLoader(HabitatTrajectoryDataset(trajectory_dir, num_steps, success_only=success_only), batch_size=batch_size,
                      num_workers=num_workers, shuffle=shuffle)

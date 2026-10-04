"""Transition windows for smoke trajectories; not a LeWM training adapter."""
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


def validate_trajectory(traj):
    if traj.get("schema_version") != 1:
        raise ValueError("Unsupported trajectory: recollect with scripts/smoke_habitat.py")
    actions = traj["actions"]
    n = len(actions)
    if actions.dtype != torch.int64 or actions.ndim != 1 or ((actions < 0) | (actions > 3)).any():
        raise ValueError("Expected int64 discrete action IDs 0..3")
    for key in ("observations", "positions", "headings", "relative_goals"):
        if len(traj[key]) != n + 1:
            raise ValueError(f"{key} must contain T+1 states for T actions")
    if len(traj["collisions"]) != n:
        raise ValueError("collisions must contain T entries")
    if traj["action_names"] != ["FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]:
        raise ValueError("Unexpected action vocabulary")


class HabitatTrajectoryDataset(Dataset):
    """num_steps consecutive transitions, with explicit current/next state fields."""
    def __init__(self, trajectory_dir, num_steps=1):
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


def create_habitat_dataloader(trajectory_dir, batch_size=1, num_steps=1, num_workers=0, shuffle=False):
    return DataLoader(HabitatTrajectoryDataset(trajectory_dir, num_steps), batch_size=batch_size,
                      num_workers=num_workers, shuffle=shuffle)

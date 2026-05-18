import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class HabitatTrajectoryDataset(Dataset):
    """
    PyTorch dataset for loading Habitat trajectories saved by habitat_collector.py.
    
    Matches LeWM's expected data format:
    - observations: (T, 3, H, W) RGB frames
    - actions: (T, 2) continuous actions (linear, angular)
    - coordinates: (T, 3) agent positions (x, z, yaw)
    - goal_coord: (3,) goal position
    - goal_image: (3, H, W) goal image (Phase 2)
    
    Args:
        trajectory_dir: Directory containing .pt trajectory files
        img_size: Target image size (default 224)
        history_len: History length for observation stacking (default 3)
        num_steps: Number of steps per sequence (history_len + predictions)
        transform: Optional transforms to apply to observations
    """
    
    def __init__(
        self,
        trajectory_dir: str,
        img_size: int = 224,
        history_len: int = 3,
        num_steps: int = 4,  # history_len + 1 prediction
        transform=None,
    ):
        self.trajectory_dir = Path(trajectory_dir)
        self.img_size = img_size
        self.history_len = history_len
        self.num_steps = num_steps
        self.transform = transform
        
        # Find all trajectory files
        self.traj_files = sorted(self.trajectory_dir.glob("traj_*.pt"))
        
        if len(self.traj_files) == 0:
            raise ValueError(f"No trajectory files found in {trajectory_dir}")
        
        # Build index of valid subtrajectories
        self.indices = []
        for traj_idx, traj_path in enumerate(self.traj_files):
            try:
                traj = torch.load(traj_path, weights_only=False)
                T = traj["observations"].shape[0]
                # Extract all valid subtrajectories of length num_steps
                for start in range(T - num_steps + 1):
                    self.indices.append((traj_idx, start))
            except Exception as e:
                print(f"Warning: Could not load {traj_path}: {e}")
        
        print(f"Loaded {len(self.traj_files)} trajectories, {len(self.indices)} subtrajectories")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_idx, start = self.indices[idx]
        traj_path = self.traj_files[traj_idx]
        traj = torch.load(traj_path, weights_only=False)
        
        # Extract subtrajectory
        obs = traj["observations"][start:start + self.num_steps]  # (T, 3, H, W)
        actions = traj["actions"][start:start + self.num_steps]    # (T, 2)
        coords = traj["coordinates"][start:start + self.num_steps] # (T, 3)
        
        # Get goal information
        goal_coord = traj["goal_coord"]  # (3,)
        goal_image = traj.get("goal_image", obs[-1])  # (3, H, W)
        
        # Apply transforms if provided
        if self.transform is not None:
            obs = self.transform(obs)
        
        # Format for LeWM: stack history frames
        # Output format matches LeWM's expected batch structure
        item = {
            "pixels": obs,           # (T, 3, H, W)
            "action": actions,       # (T, 2)
            "coordinates": coords,   # (T, 3)
            "goal_coord": goal_coord,
            "goal_image": goal_image,
        }
        
        return item
    
    def get_dim(self, key: str) -> int:
        """Get dimension of a data field."""
        if key == "action":
            return 2
        elif key == "coordinates":
            return 3
        elif key == "pixels":
            return 3 * self.img_size * self.img_size
        else:
            raise ValueError(f"Unknown key: {key}")


def create_habitat_dataloader(
    trajectory_dir: str,
    batch_size: int = 128,
    img_size: int = 224,
    history_len: int = 3,
    num_steps: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
):
    """
    Create a DataLoader for Habitat trajectories.
    
    Args:
        trajectory_dir: Directory containing .pt trajectory files
        batch_size: Batch size
        img_size: Target image size
        history_len: History length
        num_steps: Number of steps per sequence
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments passed to Dataset
    
    Returns:
        DataLoader instance
    """
    dataset = HabitatTrajectoryDataset(
        trajectory_dir=trajectory_dir,
        img_size=img_size,
        history_len=history_len,
        num_steps=num_steps,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    
    return dataloader

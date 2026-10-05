"""Small shared-cell projection + goal-conditioned behavior cloning policy."""
import torch
from torch import nn


class DirectPolicy(nn.Module):
    def __init__(self, feature_dim=1024, cells=32, projection_dim=32, hidden_dim=128, goal_only=False):
        super().__init__()
        self.goal_only = goal_only
        self.goal = nn.Sequential(nn.Linear(3, 32), nn.GELU())
        if not goal_only:
            self.visual = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, projection_dim), nn.GELU())
        self.head = nn.Sequential(nn.Linear(32 if goal_only else cells * projection_dim + 32, hidden_dim),
                                  nn.GELU(), nn.Linear(hidden_dim, 4))
        self.register_buffer('goal_mean', torch.zeros(3))
        self.register_buffer('goal_std', torch.ones(3))

    def forward(self, features, relative_goal):
        goal = self.goal((relative_goal - self.goal_mean) / self.goal_std)
        x = goal if self.goal_only else torch.cat([self.visual(features).flatten(1), goal], dim=1)
        return self.head(x)

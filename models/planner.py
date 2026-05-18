import torch
import torch.nn as nn

class CoordGoalEncoder(nn.Module):
    """
    Maps (x, y, yaw) to latent space. Matches LeWM's 192-dim [CLS] projection.
    
    Used for Phase 1 coordinate goal conditioning.
    """
    def __init__(self, latent_dim=192):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.LayerNorm(latent_dim)  # Use LayerNorm instead of BatchNorm for stability
        )

    def forward(self, coords):
        return self.mlp(coords)


def get_goal_embedding(encoder, coord_encoder, goal_img=None, goal_coord=None, device="cuda"):
    """
    Get goal embedding for planning.
    
    Phase 1: coordinate goals only.
    Phase 2: uncomment image branch for image goal support.
    
    Args:
        encoder: Visual encoder for image goals
        coord_encoder: CoordGoalEncoder for coordinate goals
        goal_img: Optional goal image tensor
        goal_coord: Optional coordinate goal tensor (x, z, yaw)
        device: Device to run on
    
    Returns:
        Goal embedding tensor of shape (B, latent_dim)
    """
    if goal_coord is not None:
        if goal_coord.dim() == 1:
            goal_coord = goal_coord.unsqueeze(0)
        return coord_encoder(goal_coord.to(device))
    elif goal_img is not None:
        # Phase 2: z_g = encoder(goal_img.unsqueeze(0).to(device))[:, 0]
        raise NotImplementedError("Image goals enabled in Phase 2 only.")
    else:
        raise ValueError("Provide either goal_coord or goal_img")

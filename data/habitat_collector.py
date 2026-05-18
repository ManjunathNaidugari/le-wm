import os
import torch
import numpy as np

def collect_habitat_episodes(
    config_path: str,
    num_episodes: int = 10000,
    max_steps: int = 200,
    frame_skip: int = 5,
    save_dir: str = "data/habitat_trajectories",
    seed: int = 42
):
    """
    Collects offline trajectories using a shortest-path heuristic + Gaussian velocity noise.
    Saves RGB frames, continuous actions, and coordinate goals in LeWM-compatible format.
    
    Note: This is a stub implementation. Requires habitat-lab and habitat-sim to be installed.
    """
    try:
        import habitat
        from habitat.config.default import get_config
    except ImportError:
        print("ERROR: habitat-lab not installed. Install with: pip install habitat-lab habitat-sim")
        print("Creating dummy data for testing purposes...")
        _create_dummy_data(save_dir, num_episodes, frame_skip)
        return
    
    os.makedirs(save_dir, exist_ok=True)
    cfg = get_config(config_path)
    cfg.defrost()
    cfg.TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
    cfg.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = 0.25
    cfg.TASK_CONFIG.SIMULATOR.TURN_ANGLE = 15.0
    cfg.freeze()

    env = habitat.Env(config=cfg)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ep_idx = 0
    while ep_idx < num_episodes:
        obs = env.reset()
        rgb_seq, action_seq, coord_seq = [], [], []
        start_pos = env.sim.get_agent_state().position
        goal_pos = env.current_episode.goals[0].position

        for step in range(max_steps):
            # Shortest-path direction + noise
            agent_pos = env.sim.get_agent_state().position
            agent_heading = env.sim.get_agent_state().rotation
            dir_vec = goal_pos - agent_pos
            dist = np.linalg.norm(dir_vec)
            if dist < 0.5:
                break

            # Convert to local frame
            yaw = np.arctan2(agent_heading[1], agent_heading[0])
            local_x = dir_vec[0] * np.cos(-yaw) - dir_vec[2] * np.sin(-yaw)
            local_y = dir_vec[0] * np.sin(-yaw) + dir_vec[2] * np.cos(-yaw)
            target_yaw = np.arctan2(local_y, local_x)

            # Continuous velocity command (linear, angular)
            v_lin = np.clip(0.5 + np.random.normal(0, 0.1), 0.0, 1.0)
            v_ang = np.clip(target_yaw + np.random.normal(0, 0.2), -1.0, 1.0)
            action = np.array([v_lin, v_ang], dtype=np.float32)

            # Execute with frame skip
            for _ in range(frame_skip):
                # Map continuous to Habitat discrete for stability
                if abs(v_ang) > 0.3:
                    discrete = "TURN_LEFT" if v_ang > 0 else "TURN_RIGHT"
                else:
                    discrete = "MOVE_FORWARD"
                obs, reward, done, info = env.step(discrete)

            rgb = obs["rgb"].transpose(2, 0, 1)  # (3, H, W)
            rgb_seq.append(torch.from_numpy(rgb).float() / 255.0)
            action_seq.append(torch.from_numpy(action))
            coord_seq.append(torch.tensor([agent_pos[0], agent_pos[2], yaw], dtype=torch.float32))

            if done:
                break

        if len(rgb_seq) >= 4:  # LeWM minimum subtrajectory length
            traj = {
                "observations": torch.stack(rgb_seq),      # (T, 3, H, W)
                "actions": torch.stack(action_seq),        # (T, 2)
                "coordinates": torch.stack(coord_seq),     # (T, 3)
                "goal_coord": torch.tensor([goal_pos[0], goal_pos[2], 0.0], dtype=torch.float32),
                "goal_image": rgb_seq[-1]                  # Phase 2 stub
            }
            torch.save(traj, os.path.join(save_dir, f"traj_{ep_idx:05d}.pt"))
            ep_idx += 1

    env.close()
    print(f"Collected {ep_idx} episodes to {save_dir}")


def _create_dummy_data(save_dir: str, num_episodes: int, frame_skip: int):
    """Create dummy trajectory data for testing without Habitat."""
    os.makedirs(save_dir, exist_ok=True)
    
    for ep_idx in range(num_episodes):
        T = np.random.randint(10, 50)  # Random trajectory length
        H, W = 224, 224
        
        traj = {
            "observations": torch.rand(T, 3, H, W),          # (T, 3, H, W)
            "actions": torch.rand(T, 2) * 2 - 1,             # (T, 2) in [-1, 1]
            "coordinates": torch.rand(T, 3) * 10 - 5,        # (T, 3)
            "goal_coord": torch.tensor([0.0, 5.0, 0.0]),     # Fixed goal
            "goal_image": torch.rand(3, H, W)                # Random goal image
        }
        torch.save(traj, os.path.join(save_dir, f"traj_{ep_idx:05d}.pt"))
    
    print(f"Created {num_episodes} dummy episodes to {save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect Habitat trajectories")
    parser.add_argument("--config", type=str, default="configs/habitat_pointnav.yaml",
                        help="Path to Habitat config")
    parser.add_argument("--num_episodes", type=int, default=10000,
                        help="Number of episodes to collect")
    parser.add_argument("--save_dir", type=str, default="data/habitat_trajectories",
                        help="Directory to save trajectories")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    collect_habitat_episodes(
        config_path=args.config,
        num_episodes=args.num_episodes,
        save_dir=args.save_dir,
        seed=args.seed
    )

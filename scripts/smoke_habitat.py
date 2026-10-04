#!/usr/bin/env python3
"""Run one real indoor-navigation episode. Exit 0 success, 1 rollout failure, 2 setup error."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "data/habitat_data/scene_datasets/habitat-test-scenes/skokloster-castle.glb")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if min(args.max_steps, args.resolution, args.fps) < 1 or args.resolution % 2:
        parser.error("steps/fps must be positive; resolution must be positive and even")
    env = None
    try:
        from envs.habitat_wrapper import HabitatSmokeEnv, require_habitat
        require_habitat()
        import torch
        import imageio.v2 as imageio
        import imageio_ffmpeg
        from data.habitat_collector import collect_episode
        imageio_ffmpeg.get_ffmpeg_exe()
        video = args.output_dir / "videos/episode_0001.mp4"
        trajectory = args.output_dir / "trajectories/episode_0001.pt"
        if video.exists() or trajectory.exists():
            raise FileExistsError("Episode output already exists; choose a different --output-dir")
        env = HabitatSmokeEnv(args.scene, args.seed, args.resolution)
        episode = collect_episode(env, args.max_steps)
        episode["metadata"] = {"scene": str(args.scene.resolve()), "seed": args.seed,
                               "habitat_sim_version": "0.3.3", "forward_step_m": 0.25,
                               "turn_degrees": 15., "goal_radius_m": env.goal_radius,
                               "fps": args.fps}
        video.parent.mkdir(parents=True, exist_ok=True)
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        torch.save(episode, trajectory)
        with imageio.get_writer(str(video), fps=args.fps, codec="libx264", macro_block_size=1) as writer:
            for frame in episode["observations"]:
                writer.append_data(frame.permute(1, 2, 0).numpy())
        m = episode["metrics"]
        print(f"{'SUCCESS' if m['success'] else 'FAILURE'}: steps={m['steps']} "
              f"path_length={m['path_length']:.3f}m final_distance={m['final_distance']:.3f}m "
              f"termination={m['termination']}")
        print(f"Trajectory: {trajectory}\nVideo: {video}")
        return 0 if m["success"] else 1
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Habitat smoke test error: {exc}", file=sys.stderr)
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    sys.exit(main())

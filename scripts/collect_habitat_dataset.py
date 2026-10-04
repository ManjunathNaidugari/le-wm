#!/usr/bin/env python3
"""Collect real expert episodes; exit 0 completed, 2 collection/setup error."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['test-scene', 'gibson'], default='test-scene')
    parser.add_argument('--scene-data-dir', type=Path)
    parser.add_argument('--episode-data', type=Path)
    parser.add_argument('--num-episodes', type=int, help='Default: 20 for Gibson, 100 for test-scene')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/habitat_expert')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--scene', type=Path, default=ROOT / 'data/habitat_data/scene_datasets/habitat-test-scenes/skokloster-castle.glb')
    args = parser.parse_args()
    if args.num_episodes is None:
        args.num_episodes = 20 if args.dataset == 'gibson' else 100
    if not 1 <= args.num_episodes < 2**31 or not 0 <= args.seed < 2**31 or args.max_steps < 1 or args.resolution < 2 or args.resolution % 2:
        parser.error('invalid episode count/seed/steps; resolution must be positive and even')
    if args.dataset == 'gibson' and (args.scene_data_dir is None or args.episode_data is None):
        parser.error('Gibson requires --scene-data-dir and --episode-data')
    if args.dataset == 'test-scene' and (args.scene_data_dir or args.episode_data):
        parser.error('PointNav paths require --dataset gibson')
    env = None
    try:
        if args.output_dir.exists():
            raise FileExistsError('Output directory exists; choose a new directory (no overwrite/resume)')
        from envs.habitat_wrapper import HabitatSmokeEnv
        from data.habitat_dataset_collection import collect_dataset, summary
        if args.dataset == 'gibson':
            from data.pointnav import load_pointnav
            from data.pointnav_collection import collect_pointnav
            from envs.habitat_wrapper import require_habitat
            definitions = load_pointnav(args.episode_data, args.scene_data_dir, args.num_episodes, args.seed)
            require_habitat()
            manifest = collect_pointnav(definitions, args.output_dir, args.seed, args.max_steps, args.resolution)
        else:
            env = HabitatSmokeEnv(args.scene, args.seed, args.resolution)
            manifest = collect_dataset(env, args.output_dir, args.scene, args.num_episodes, args.seed, args.max_steps)
        print(json.dumps(summary(manifest['episodes']), indent=2))
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f'Dataset collection error: {exc}', file=sys.stderr)
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    sys.exit(main())

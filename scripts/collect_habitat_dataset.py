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
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/habitat_expert')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--scene', type=Path, default=ROOT / 'data/habitat_data/scene_datasets/habitat-test-scenes/skokloster-castle.glb')
    args = parser.parse_args()
    if not 1 <= args.num_episodes < 2**31 or not 0 <= args.seed < 2**31 or args.max_steps < 1 or args.resolution < 2 or args.resolution % 2:
        parser.error('invalid episode count/seed/steps; resolution must be positive and even')
    env = None
    try:
        if args.output_dir.exists():
            raise FileExistsError('Output directory exists; choose a new directory (no overwrite/resume)')
        from envs.habitat_wrapper import HabitatSmokeEnv
        from data.habitat_dataset_collection import collect_dataset, summary
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

#!/usr/bin/env python3
"""Validate all trajectories and print summary; optionally export one episode video."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--episode', type=int, help='Episode ID to inspect and export; omit for summary only')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1 or (args.episode is not None and args.episode < 0):
        parser.error('fps must be positive and episode ID nonnegative')
    try:
        import torch
        from data.habitat_dataset_collection import validate_dataset, summary
        manifest = validate_dataset(args.dataset_dir)
        print(json.dumps(dict(complete=manifest['complete'], **summary(manifest['episodes'])), indent=2))
        if args.episode is not None:
            entries = [e for e in manifest['episodes'] if e['episode_id'] == args.episode]
            if not entries:
                raise ValueError(f'Episode {args.episode} not in manifest')
            entry = entries[0]
            print(json.dumps(entry, indent=2))
            episode = torch.load(args.dataset_dir / entry['trajectory'], map_location='cpu', weights_only=True)
            import imageio.v2 as imageio
            video = args.dataset_dir / 'videos' / f'episode_{args.episode:06d}.mp4'
            if video.exists():
                raise FileExistsError(f'Video already exists: {video}')
            video.parent.mkdir(exist_ok=True)
            with imageio.get_writer(str(video), fps=args.fps, codec='libx264', macro_block_size=1) as writer:
                for frame in episode['observations']:
                    writer.append_data(frame.permute(1, 2, 0).numpy())
            print(f"RGB shape: {list(episode['observations'].shape)}; video: {video}")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f'Dataset inspection error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

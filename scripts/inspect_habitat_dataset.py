#!/usr/bin/env python3
"""Validate all trajectories and print summary; optionally export one episode video."""
import argparse
import json
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--episode', type=int, nargs='+', help='Episode IDs to inspect and export; omit for summary only')
    parser.add_argument('--qc', action='store_true', help='Write per-episode RGB quality report')
    parser.add_argument('--export-flagged', action='store_true', help='Export QC-flagged episodes only')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1 or (args.episode is not None and min(args.episode) < 0):
        parser.error('fps must be positive and episode ID nonnegative')
    if args.export_flagged and not args.qc:
        parser.error('--export-flagged requires --qc')
    try:
        import torch
        from data.habitat_dataset_collection import validate_dataset, summary
        selected = set(args.episode or [])
        if args.qc:
            from data.rgb_qc import qc_dataset
            report = qc_dataset(args.dataset_dir)
            print('episode scene steps brightness black saturation frozen min_y max_y qc_status')
            for row in report['episodes']:
                fields = ['episode', 'scene', 'steps', 'mean_brightness', 'black_ratio',
                          'saturation_ratio', 'motion_frozen_ratio', 'min_y', 'max_y', 'qc_status']
                print(' '.join(str(row.get(k)) for k in fields))
                if args.export_flagged and row['qc_status'] == 'flagged' and isinstance(row['episode'], int):
                    selected.add(row['episode'])
            print(f"QC flagged {report['flagged']}/{report['total']}; report: {args.dataset_dir / 'qc_report.json'}")
        manifest = validate_dataset(args.dataset_dir, check_rgb_quality=False)
        print(json.dumps(dict(complete=manifest['complete'], **summary(manifest['episodes'])), indent=2))
        print('Scene counts:', {scene: sum(e['scene'] == scene for e in manifest['episodes'])
                                for scene in sorted({e['scene'] for e in manifest['episodes']})})
        for episode_id in sorted(selected):
            entries = [e for e in manifest['episodes'] if e['episode_id'] == episode_id]
            if not entries:
                raise ValueError(f'Episode {episode_id} not in manifest')
            entry = entries[0]
            print(json.dumps(entry, indent=2))
            episode = torch.load(args.dataset_dir / entry['trajectory'], map_location='cpu', weights_only=True)
            import imageio.v2 as imageio
            video = args.dataset_dir / 'videos' / f'episode_{episode_id:06d}.mp4'
            if video.exists():
                print(f'Skipping existing video: {video}')
                continue
            video.parent.mkdir(exist_ok=True)
            with imageio.get_writer(str(video), fps=args.fps, codec='libx264', macro_block_size=1) as writer:
                for frame in episode['observations']:
                    writer.append_data(frame.permute(1, 2, 0).numpy())
            print(f"RGB shape: {list(episode['observations'].shape)}; video: {video}")
        return 0
    except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError) as exc:
        print(f'Dataset inspection error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

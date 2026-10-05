"""Explicit development scene selection, small runs, video export and comparison."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys

import torch

from jepa_navigation.data.pointnav import load_pointnav
from jepa_navigation.data.gibson_pilot import collect_pilot, compare_episodes, validate_pilot
from .config import SimulatorConfig, NavigationConfig, load_config
from .video import export_video


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for mode in ('check', 'run'):
        p = commands.add_parser(mode)
        p.add_argument('--episode-data', type=Path, required=True,
                       help='Official train index or a selected train/content/building.json.gz')
        p.add_argument('--scene-data-dir', type=Path, required=True)
        p.add_argument('--scenes', nargs='+', required=True, help='1–3 bare building names; choose train buildings')
        p.add_argument('--episode-id', action='append', help='Exact source ID; select one building for an exact single episode')
        p.add_argument('--num-episodes', type=int, default=1)
        p.add_argument('--simulator-config', type=Path, default=Path('configs/simulator.yaml'))
        p.add_argument('--navigation-config', type=Path, default=Path('configs/gibson_pilot.yaml'))
        p.add_argument('--seed', type=int)
        p.add_argument('--max-steps', type=int)
        p.add_argument('--output-dir', type=Path, required=mode == 'run')
    p = commands.add_parser('export')
    p.add_argument('trajectory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=int, default=10)
    p = commands.add_parser('compare')
    p.add_argument('first', type=Path)
    p.add_argument('second', type=Path)
    p.add_argument('--atol', type=float, default=1e-6)
    p = commands.add_parser('validate')
    p.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'compare':
            report = compare_episodes(*(torch.load(p, map_location='cpu', weights_only=True)
                                        for p in (args.first, args.second)), atol=args.atol)
            print(json.dumps(report, indent=2))
            return 0 if report['repeatable'] else 1
        if args.command == 'export':
            episode = torch.load(args.trajectory, map_location='cpu', weights_only=True)
            validate_pilot(episode)
            print(export_video(episode, args.output, args.fps, composite=True))
            return 0
        if args.command == 'validate':
            from jepa_navigation.data.gibson_pilot import validate_pilot_run
            manifest = validate_pilot_run(args.run_dir)
            print(json.dumps(manifest['summary'], indent=2))
            return 0 if manifest['complete'] else 1
        if not 1 <= args.num_episodes <= 20 or not 1 <= len(set(args.scenes)) <= 3:
            raise ValueError('Select 1–20 episodes across 1–3 development buildings')
        # Final evaluation stays separate; no accidental val/test pilot through this command.
        parts = {part.lower() for part in args.episode_data.resolve().parts}
        if 'train' not in parts or parts.intersection({'val', 'test', 'val_mini'}):
            raise ValueError('Use official train/ episodes for development; reserve val/test for final evaluation')
        if args.episode_id and len(set(args.scenes)) != 1:
            raise ValueError('Exact --episode-id selection requires one building (IDs are scene-local)')
        sim = load_config(args.simulator_config, SimulatorConfig)
        nav = load_config(args.navigation_config, NavigationConfig,
                          dict(seed=args.seed, max_steps=args.max_steps,
                               episode_data=args.episode_data, scene_data_dir=args.scene_data_dir,
                               output_dir=args.output_dir))
        definitions = load_pointnav(args.episode_data, args.scene_data_dir, args.num_episodes, nav.seed,
                                    scenes=args.scenes, episode_ids=args.episode_id, require_scenes=False)
        if args.command == 'check':
            ready = all(Path(d['scene']).is_file() for d in definitions)
            runtime = {name: importlib.util.find_spec(name) is not None for name in ('habitat_sim', 'habitat')}
            print(json.dumps(dict(platform=platform.platform(), runtime_available=runtime,
                                  selected=[dict(scene=d['scene'], scene_exists=Path(d['scene']).is_file(),
                                                 source_episode_id=d['source_episode_id'],
                                                 source_file=d['source_episode_file']) for d in definitions]), indent=2))
            return 0 if ready and all(runtime.values()) else 2
        manifest = collect_pilot(definitions, args.output_dir, sim, nav)
        print(json.dumps(manifest['summary'], indent=2))
        entries = manifest['episodes']
        if any(e['status'] != 'finished' for e in entries):
            return 2
        return 0 if all(e['success'] for e in entries) else 1
    except (ImportError, OSError, RuntimeError, ValueError, AssertionError, KeyError) as exc:
        print(f'Gibson pilot error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

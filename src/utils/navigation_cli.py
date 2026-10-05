"""CLI orchestration for the installed simulator foundation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .config import SimulatorConfig, NavigationConfig, load_config


def parser_for(description, collection=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--simulator-config', type=Path)
    parser.add_argument('--navigation-config', type=Path)
    parser.add_argument('--scene-data-dir', type=Path)
    parser.add_argument('--episode-data', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--fps', type=int)
    if collection:
        parser.add_argument('--num-episodes', type=int, default=20)
        parser.add_argument('--dataset', choices=['gibson', 'test-scene'], default='gibson')
        parser.add_argument('--scene', type=Path, help='Required only for optional test-scene compatibility mode')
    return parser


def settings(args):
    sim = load_config(args.simulator_config, SimulatorConfig, dict(resolution=args.resolution))
    nav = load_config(args.navigation_config, NavigationConfig,
        {key: getattr(args, key) for key in ('scene_data_dir', 'episode_data', 'output_dir',
                                            'seed', 'max_steps', 'fps')})
    return sim, nav


def definitions(nav, count):
    from jepa_navigation.data.pointnav import load_pointnav
    return load_pointnav(Path(nav.episode_data).expanduser(), Path(nav.scene_data_dir).expanduser(), count, nav.seed)


def show_metrics(episode):
    print(json.dumps(episode['metrics'], indent=2))


def execute(mode):
    parser = parser_for({'verify': 'Verify Habitat, Gibson assets, navmesh and RGB camera.',
                         'run': 'Run one official Gibson episode without saving.',
                         'record': 'Save one official Gibson trajectory and MP4.',
                         'collect': 'Collect official Gibson expert trajectories.'}[mode], mode == 'collect')
    args = parser.parse_args()
    try:
        sim, nav = settings(args)
        from jepa_navigation.simulator.habitat_env import require_habitat
        require_habitat()
        if mode == 'collect' and (args.num_episodes < 1 or args.num_episodes >= 2**31):
            raise ValueError('num-episodes must be in [1, 2**31)')
        if mode == 'collect' and args.dataset == 'test-scene':
            if args.scene is None:
                raise ValueError('--scene is required for test-scene mode')
            from jepa_navigation.simulator.habitat_env import HabitatSmokeEnv
            from jepa_navigation.data.habitat_dataset_collection import collect_dataset, summary
            env = HabitatSmokeEnv(args.scene, nav.seed, sim.resolution,
                                  simulator_config=sim, navigation_config=nav)
            try:
                manifest = collect_dataset(env, Path(nav.output_dir).expanduser(), args.scene,
                                           args.num_episodes, nav.seed, nav.max_steps)
            finally:
                env.close()
            print(json.dumps(summary(manifest['episodes']), indent=2))
            return 0
        selected = definitions(nav, args.num_episodes if mode == 'collect' else 1)
        if mode == 'verify':
            from jepa_navigation.simulator.pointnav_env import PointNavEnv
            import imageio_ffmpeg
            import numpy as np
            env = PointNavEnv(selected[0], nav.seed, sim.resolution,
                              simulator_config=sim, navigation_config=nav)
            try:
                initial = env.reset()
                if initial['rgb'].shape != (sim.resolution, sim.resolution, 3) or initial['rgb'].dtype != np.uint8:
                    raise RuntimeError('Unexpected RGB shape/dtype')
                print(f"Habitat-Sim 0.3.3 ready; scene={selected[0]['scene']}; RGB={initial['rgb'].shape}")
                print(f'Initial geodesic distance: {env.initial_distance:.3f}m')
                print(f'FFmpeg: {imageio_ffmpeg.get_ffmpeg_exe()}')
            finally:
                env.close()
            return 0
        from jepa_navigation.data.pointnav_collection import run_official_episode, collect_pointnav
        if mode == 'run':
            episode = run_official_episode(selected[0], nav.seed, nav.max_steps, sim.resolution,
                                           sim, nav)
            show_metrics(episode)
            return 0 if episode['metrics']['success'] else 1
        root = Path(nav.output_dir).expanduser()
        if root.exists():
            raise FileExistsError(f'Choose a new output directory: {root}')
        directory = root / 'trajectories' if mode == 'record' else root
        manifest = collect_pointnav(selected, directory, nav.seed, nav.max_steps, sim.resolution, sim, nav)
        from jepa_navigation.data.habitat_dataset_collection import summary, validate_dataset
        validate_dataset(directory, check_rgb_quality=False)
        print(json.dumps(summary(manifest['episodes']), indent=2))
        if mode == 'record':
            import torch
            from .video import export_video
            path = directory / 'episode_000000.pt'
            episode = torch.load(path, map_location='cpu', weights_only=True)
            video = export_video(episode, root / 'videos/episode_000000.mp4', nav.fps)
            print(f'Trajectory: {path}\nVideo: {video}')
            return 0 if episode['metrics']['success'] else 1
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f'Navigation {mode} error: {exc}', file=sys.stderr)
        return 2


def verify_main():
    return execute('verify')


def run_main():
    return execute('run')


def record_main():
    return execute('record')


def collect_main():
    return execute('collect')

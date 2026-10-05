"""Official PointNav execution and collection using the stable T+1/T contract."""
from dataclasses import replace
from pathlib import Path
import torch

from .habitat_collector import collect_episode
from .habitat_dataset_collection import episode_seed, manifest_entry, write_manifest
from .habitat_dataset import validate_trajectory
from jepa_navigation.simulator.pointnav_env import PointNavEnv
from jepa_navigation.utils.config import SimulatorConfig, NavigationConfig, effective_settings


def run_official_episode(definition, seed=42, max_steps=500, resolution=224,
                         simulator_config=None, navigation_config=None, episode_id=0):
    sim = simulator_config or SimulatorConfig(resolution=resolution)
    nav = replace(navigation_config or NavigationConfig(), seed=seed, max_steps=max_steps)
    # Preserve the established constructor API for callers using defaults.
    kwargs = {} if simulator_config is None and navigation_config is None else dict(
        simulator_config=sim, navigation_config=nav)
    env = PointNavEnv(definition, seed, sim.resolution, **kwargs)
    try:
        episode = collect_episode(env, max_steps)
        nav = replace(nav, goal_radius_m=env.goal_radius)
        episode['metadata'] = dict(definition, seed=seed, episode_id=episode_id,
            habitat_sim_version='0.3.3', forward_step_m=nav.forward_step_m,
            turn_degrees=nav.turn_degrees, goal_radius_m=env.goal_radius,
            agent_height_m=sim.agent_height_m, agent_radius_m=sim.agent_radius_m,
            navmesh='recomputed_for_agent', camera=dict(uuid='rgb', type='COLOR',
            resolution=[sim.resolution, sim.resolution], position=[0., sim.camera_height_m, 0.],
            hfov_degrees=sim.hfov_degrees), effective_settings=effective_settings(sim, nav))
        validate_trajectory(episode, check_rgb_quality=False)
        return episode
    finally:
        env.close()


def collect_pointnav(definitions, output_dir, seed=42, max_steps=500, resolution=224,
                     simulator_config=None, navigation_config=None):
    if not definitions or max_steps < 1 or resolution < 2 or resolution % 2:
        raise ValueError('Need episodes, positive max_steps and a positive even resolution')
    episode_seed(seed, len(definitions)-1)
    identities = [(d['source_scene_id'], d['source_episode_id']) for d in definitions]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate official scene/episode IDs')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, source_dataset='gibson_pointnav_v1', global_seed=seed,
                    requested_episodes=len(definitions), max_steps=max_steps,
                    complete=False, episodes=[])
    write_manifest(root, manifest)
    for index, definition in enumerate(definitions):
        episode = run_official_episode(definition, episode_seed(seed, index), max_steps, resolution,
                                       simulator_config, navigation_config, index)
        filename = f'episode_{index:06d}.pt'
        tmp = root / (filename + '.tmp')
        torch.save(episode, tmp)
        tmp.replace(root / filename)
        entry = manifest_entry(index, episode, filename)
        manifest['episodes'].append(entry)
        write_manifest(root, manifest)
        print(f"episode={index:06d} scene={definition['source_scene_id']} "
              f"source_episode={definition['source_episode_id']} success={entry['success']} "
              f"steps={entry['steps']}", flush=True)
    manifest['complete'] = True
    write_manifest(root, manifest)
    return manifest

"""Small official PointNav collection; the smoke collector remains unchanged."""
from pathlib import Path
import torch

from data.habitat_collector import collect_episode
from data.habitat_dataset_collection import episode_seed, manifest_entry, write_manifest
from datasets.habitat_dataset import validate_trajectory
from envs.pointnav_wrapper import PointNavEnv


def collect_pointnav(definitions, output_dir, seed=42, max_steps=500, resolution=224):
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
        current_seed = episode_seed(seed, index)
        # One scene at a time; close even on collection/validation errors.
        env = PointNavEnv(definition, current_seed, resolution)
        try:
            episode = collect_episode(env, max_steps)
            episode['metadata'] = dict(definition, seed=current_seed, episode_id=index,
                habitat_sim_version='0.3.3', forward_step_m=0.25, turn_degrees=15.,
                goal_radius_m=env.goal_radius, agent_height_m=1.5, agent_radius_m=0.1,
                navmesh='recomputed_for_agent', camera=dict(uuid='rgb', type='COLOR',
                resolution=[resolution, resolution], position=[0., 1.5, 0.], hfov_degrees=90.))
            validate_trajectory(episode, check_rgb_quality=False)
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
        finally:
            env.close()
    manifest['complete'] = True
    write_manifest(root, manifest)
    return manifest

"""Small manifest-backed dataset collection and validation helpers."""
import json
import math
from pathlib import Path
from statistics import mean, median

import torch
from data.habitat_collector import collect_episode
from datasets.habitat_dataset import validate_trajectory


def episode_seed(seed, episode_id):
    """Unique signed-32-bit seeds; episode zero preserves the smoke-test seed."""
    if not 0 <= seed < 2**31 or not 0 <= episode_id < 2**31:
        raise ValueError("seed and episode ID must be in [0, 2**31)")
    return (seed + episode_id) % (2**31)


def manifest_entry(episode_id, episode, filepath):
    m, meta = episode['metrics'], episode['metadata']
    entry = dict(episode_id=episode_id, scene=meta['scene'], seed=meta['seed'],
                success=m['success'], steps=m['steps'], termination=m['termination'],
                initial_geodesic_distance=m['initial_geodesic_distance'],
                final_distance=m['final_distance'] if math.isfinite(m['final_distance']) else None,
                path_length=m['path_length'], collision_count=int(episode['collisions'].sum()),
                trajectory=filepath)
    if meta.get('source_dataset'):
        entry.update(source_dataset=meta['source_dataset'],
                     source_episode_id=meta['source_episode_id'], source_scene_id=meta['source_scene_id'])
    return entry


def summary(entries):
    successes = sum(e['success'] for e in entries)
    return dict(total_episodes=len(entries), successes=successes,
                failures=len(entries) - successes,
                success_rate=successes / len(entries) if entries else 0.,
                mean_episode_length=mean(e['steps'] for e in entries) if entries else 0.,
                median_episode_length=median(e['steps'] for e in entries) if entries else 0.,
                mean_initial_goal_distance=mean(e['initial_geodesic_distance'] for e in entries) if entries else 0.,
                mean_collisions=mean(e['collision_count'] for e in entries) if entries else 0.)


def write_manifest(root, manifest):
    tmp = root / 'manifest.json.tmp'
    tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    tmp.replace(root / 'manifest.json')


def collect_dataset(env, output_dir, scene, num_episodes=100, seed=42, max_steps=500):
    """Reuse one simulator/navmesh, reseeding both random generators per episode.

    Refuse existing directories. Commit each manifest entry after its trajectory;
    interrupted runs remain inspectable, but automatic resume is intentionally absent.
    """
    if not 1 <= num_episodes < 2**31 or max_steps < 1:
        raise ValueError('num_episodes and max_steps must be positive')
    episode_seed(seed, num_episodes - 1)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, global_seed=seed, requested_episodes=num_episodes,
                    max_steps=max_steps, complete=False, episodes=[])
    write_manifest(root, manifest)
    seen = set()
    for episode_id in range(num_episodes):
        current_seed = episode_seed(seed, episode_id)
        env.seed = current_seed
        env.sim.seed(current_seed)
        env.sim.pathfinder.seed(current_seed)
        episode = collect_episode(env, max_steps)
        episode['metadata'] = dict(scene=str(Path(scene).resolve()), seed=current_seed,
                                   episode_id=episode_id, habitat_sim_version='0.3.3',
                                   forward_step_m=0.25, turn_degrees=15., goal_radius_m=env.goal_radius)
        validate_trajectory(episode)
        # Detect identical sampled tasks, rather than silently collecting duplicates.
        signature = tuple(episode['positions'][0].tolist() + episode['goal_position'].tolist()
                          + [episode['headings'][0].item()])
        if signature in seen:
            raise RuntimeError(f'Duplicate start/goal/heading at episode {episode_id}; collection stopped')
        seen.add(signature)
        filename = f'episode_{episode_id:06d}.pt'
        tmp = root / (filename + '.tmp')
        torch.save(episode, tmp)
        tmp.replace(root / filename)
        manifest['episodes'].append(manifest_entry(episode_id, episode, filename))
        write_manifest(root, manifest)
        m = episode['metrics']
        print(f"episode={episode_id:06d} seed={current_seed} success={m['success']} "
              f"steps={m['steps']} termination={m['termination']}", flush=True)
    manifest['complete'] = True
    write_manifest(root, manifest)
    return manifest


def validate_dataset(dataset_dir, check_rgb_quality=True):
    """Read every saved tensor and cross-check manifest metadata; never trust stale stats."""
    root = Path(dataset_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('episodes'), list):
        raise ValueError('Unsupported dataset manifest')
    seen = set()
    paths = set()
    for expected_id, entry in enumerate(manifest['episodes']):
        filename = f'episode_{expected_id:06d}.pt'
        if entry['episode_id'] != expected_id or entry['trajectory'] != filename:
            raise ValueError('Manifest episode IDs/filenames must be consecutive and unique')
        episode = torch.load(root / filename, map_location='cpu', weights_only=True)
        validate_trajectory(episode, check_rgb_quality=check_rgb_quality)
        if episode['metadata']['seed'] != episode_seed(manifest['global_seed'], expected_id):
            raise ValueError(f'{filename}: seed mismatch')
        if episode['metadata']['episode_id'] != expected_id:
            raise ValueError(f'{filename}: episode ID mismatch')
        if entry != manifest_entry(expected_id, episode, filename):
            raise ValueError(f'{filename}: manifest does not match trajectory')
        signature = (episode['metadata']['scene'],) + tuple(episode['positions'][0].tolist() + episode['goal_position'].tolist()
                          + [episode['headings'][0].item()])
        if episode['metadata'].get('source_dataset') == 'gibson_pointnav_v1':
            signature = (episode['metadata']['source_scene_id'], episode['metadata']['source_episode_id'])
        if signature in seen:
            raise ValueError(f'{filename}: duplicate sampled task or official episode')
        seen.add(signature)
        paths.add(filename)
    if paths != {p.name for p in root.glob('*.pt')}:
        raise ValueError('Unlisted trajectory files in dataset directory')
    if len(paths) > manifest['requested_episodes'] or (
            manifest['complete'] and len(paths) != manifest['requested_episodes']):
        raise ValueError('Manifest episode count mismatch')
    return manifest

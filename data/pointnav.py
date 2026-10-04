"""Official PointNav-v1 JSON ingestion without a Habitat-Lab runtime dependency."""
import glob
import gzip
import hashlib
import json
from pathlib import Path
import random
import string

import numpy as np


def _read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or not isinstance(document.get('episodes'), list):
        raise ValueError(f'{path}: expected a PointNav episodes list')
    return document


def _vector(value, length, field):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid {field}') from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f'{field} must have {length} finite numbers')
    return array.tolist()


def resolve_scene(scene_id, scene_data_dir):
    """Same data/scene_datasets prefix removal as Habitat-Lab, no basename guessing."""
    if not isinstance(scene_id, str) or not scene_id:
        raise ValueError('Missing scene_id')
    prefix = 'data/scene_datasets/'
    relative = scene_id[len(prefix):] if scene_id.startswith(prefix) else scene_id
    root = Path(scene_data_dir).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or root not in path.parents:
        raise ValueError(f'Scene must resolve inside --scene-data-dir: {scene_id}')
    if path.suffix != '.glb' or not path.is_file():
        raise FileNotFoundError(f'Gibson scene missing: {path}')
    return path


def parse_episode(raw, scene_data_dir, source_file, source_sha256):
    if not isinstance(raw, dict) or not isinstance(raw.get('episode_id'), (str, int)):
        raise ValueError('PointNav episode requires episode_id')
    scene = resolve_scene(raw.get('scene_id'), scene_data_dir)
    start = _vector(raw.get('start_position'), 3, 'start_position')
    rotation = _vector(raw.get('start_rotation'), 4, 'start_rotation XYZW')
    if not np.isclose(np.linalg.norm(rotation), 1., atol=1e-4, rtol=0):
        raise ValueError('start_rotation must be a unit quaternion; refusing to alter pose')
    goals = raw.get('goals')
    if not isinstance(goals, list) or len(goals) != 1 or not isinstance(goals[0], dict):
        raise ValueError('This PointNav collector supports exactly one coordinate goal')
    goal = _vector(goals[0].get('position'), 3, 'goal position')
    radius = goals[0].get('radius')
    if radius is not None and (not isinstance(radius, (float, int)) or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Invalid goal radius')
    return dict(source_dataset='gibson_pointnav_v1', source_scene_id=raw['scene_id'],
                source_episode_id=str(raw['episode_id']), scene=str(scene),
                start_position=start, start_rotation_xyzw=rotation, goal_position=goal,
                source_goal_radius=radius, source_episode_file=str(Path(source_file).resolve()),
                source_episode_sha256=source_sha256)


def load_pointnav(episode_data, scene_data_dir, num_episodes, seed=42):
    """Read monolithic JSON or an official split index plus content shards.

    Bound retained candidates to N per scene; reservoir sample deterministically,
    then round-robin scenes to expose scene variation in a small validation batch.
    Each JSON shard is decoded separately; no environment or asset download occurs.
    """
    if num_episodes < 1:
        raise ValueError('num_episodes must be positive')
    index = Path(episode_data).resolve()
    header = _read(index)
    template = header.get('content_scenes_path', '{data_path}/content/{scene}.json.gz')
    fields = [field for _, field, _, _ in string.Formatter().parse(template) if field is not None]
    if set(fields) - {'data_path', 'scene'} or fields.count('scene') != 1:
        raise ValueError('Unsupported content_scenes_path template')
    pattern = template.format(data_path=str(index.parent), scene='*')
    shards = sorted(Path(p).resolve() for p in glob.glob(pattern))
    if any(index.parent not in p.parents for p in shards):
        raise ValueError('Content shards must be inside the episode split directory')
    rng = random.Random(seed)
    pools, counts = {}, {}
    for source in [index] + [p for p in shards if p != index]:
        doc = header if source == index else _read(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        for raw in doc['episodes']:
            if not isinstance(raw, dict) or not isinstance(raw.get('scene_id'), str):
                raise ValueError(f'{source}: invalid episode scene_id')
            key = raw['scene_id']
            count = counts.get(key, 0) + 1
            counts[key] = count
            pool = pools.setdefault(key, [])
            candidate = (raw, source, digest)
            if len(pool) < num_episodes:
                pool.append(candidate)
            else:
                slot = rng.randrange(count)
                if slot < num_episodes:
                    pool[slot] = candidate
        if source == index:
            # Release potentially large monolithic header after consuming it.
            header = None
    scene_ids = sorted(pools)
    rng.shuffle(scene_ids)
    for pool in pools.values():
        rng.shuffle(pool)
    selected = []
    while len(selected) < num_episodes:
        added = False
        for scene_id in scene_ids:
            if pools[scene_id] and len(selected) < num_episodes:
                raw, source, digest = pools[scene_id].pop()
                selected.append(parse_episode(raw, scene_data_dir, source, digest))
                added = True
        if not added:
            raise ValueError(f'Requested {num_episodes} episodes but found only {len(selected)}')
    keys = [(e['source_scene_id'], e['source_episode_id']) for e in selected]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate scene/episode IDs in selected PointNav definitions')
    return selected

"""Small, auditable PointNav pilot. Failures stay attached to their source IDs."""
from dataclasses import replace
import json
import hashlib
import importlib.metadata
import platform
from pathlib import Path
import time

import numpy as np
import torch

from .habitat_collector import collect_episode
from .habitat_dataset import validate_trajectory
from .habitat_dataset_collection import manifest_entry, write_manifest
from jepa_navigation.simulator.lab_pointnav_env import LabPointNavEnv
from jepa_navigation.utils.config import effective_settings
from jepa_navigation.utils.video import export_video


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_pilot(episode):
    validate_trajectory(episode)
    meta, m = episode['metadata'], episode['metrics']
    q = episode['rotations_xyzw']
    if q.shape != (len(episode['actions']) + 1, 4) or not torch.isfinite(q).all():
        raise ValueError('rotations_xyzw must contain T+1 finite XYZW quaternions')
    if not torch.allclose(torch.linalg.vector_norm(q, dim=1), torch.ones(len(q)), atol=1e-4, rtol=0):
        raise ValueError('Recorded rotations must be unit quaternions')
    np.testing.assert_allclose(episode['positions'][0], meta['start_position'], atol=1e-5, rtol=0)
    np.testing.assert_allclose(episode['goal_position'], meta['goal_position'], atol=1e-5, rtol=0)
    expected = np.asarray(meta['start_rotation_xyzw'])
    initial = q[0].numpy()
    if min(np.max(np.abs(initial - expected)), np.max(np.abs(initial + expected))) > 1e-5:
        raise ValueError('Recorded start rotation differs from official definition')
    # Rotate world deltas by inverse quaternion, independently of Habitat geometry.
    delta = episode['goal_position'].numpy() - episode['positions'].numpy()
    xyz, w = -q[:, :3].numpy(), q[:, 3:].numpy()
    t = 2 * np.cross(xyz, delta)
    local = delta + w * t + np.cross(xyz, t)
    relative = np.stack((-local[:, 2], -local[:, 0], local[:, 1]), axis=1)
    np.testing.assert_allclose(episode['relative_goals'], relative, atol=1e-4, rtol=1e-5)
    expected_success = m['termination'] == 'stop' and m['final_distance'] < meta['goal_radius_m']
    if m['success'] != expected_success or bool(m['habitat_lab']['success']) != expected_success:
        raise ValueError('Success differs from Habitat-Lab strict STOP + distance criterion')
    stop = (episode['actions'] == 3).nonzero().flatten().tolist()
    if stop:
        for field in ('positions', 'rotations_xyzw', 'observations'):
            if not torch.equal(episode[field][-1], episode[field][-2]):
                raise ValueError(f'STOP changed {field}; expected a terminal no-op')
    travelled = float(torch.linalg.vector_norm(torch.diff(episode['positions'], dim=0), dim=1).sum())
    if not np.isclose(travelled, m['path_length'], atol=1e-4):
        raise ValueError('Distance travelled disagrees with recorded poses')
    frames = episode['visualization_frames']
    if frames.dtype != torch.uint8 or frames.ndim != 4 or frames.shape[0] != len(q) or frames.shape[-1] != 3:
        raise ValueError('Expected T+1 uint8 Habitat composite frames')


def run_pilot_episode(definition, sim, nav, controller=None):
    env = LabPointNavEnv(definition, sim, nav)
    try:
        episode = collect_episode(env, nav.max_steps, controller=controller)
        nav = replace(nav, goal_radius_m=env.goal_radius)
        packages = {}
        for name in ('habitat-lab', 'numpy', 'torch', 'hydra-core', 'omegaconf', 'imageio', 'imageio-ffmpeg'):
            packages[name] = importlib.metadata.version(name)
        episode['metadata'] = dict(
            definition, seed=nav.seed, episode_id=0, workflow='habitat_lab_pointnav_reference',
            habitat_sim_version=env.hs.__version__, habitat_lab_version='0.3.3',
            forward_step_m=nav.forward_step_m, turn_degrees=nav.turn_degrees,
            goal_radius_m=env.goal_radius, success_comparison='strict_less_than',
            coordinate_convention='Y-up; yaw=0 faces -Z; relative=(forward,left,up)',
            action_conversion={'lab_to_recorded': [3, 0, 1, 2], 'recorded_to_lab': [1, 2, 3, 0]},
            navmesh='Habitat-Lab default_agent_navmesh matches configured agent',
            effective_settings=effective_settings(sim, nav),
            effective_lab_config=env.effective_lab_config,
            packages=packages, platform=platform.platform(),
            controller='ShortestPathFollower(stop_on_error=False)' if controller is None else 'supplied_policy',
        )
        validate_pilot(episode)
        return episode
    finally:
        env.close()


def collect_pilot(definitions, output_dir, sim, nav, *, runner=None):
    if not definitions or len(definitions) > 20:
        raise ValueError('Pilot must contain 1–20 episodes')
    keys = [(d['source_scene_id'], d['source_episode_id']) for d in definitions]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate official scene/episode IDs')
    if len({key[0] for key in keys}) > 3:
        raise ValueError('Pilot supports at most three development buildings')
    runner = runner or run_pilot_episode
    root = Path(output_dir).expanduser().resolve()
    # Avoid accidentally placing run artifacts under the implementation directories.
    source = Path(__file__).resolve().parents[2]
    if root == source or any(folder == root or folder in root.parents for folder in
                             (source / name for name in ('src', 'scripts', 'tests', 'docs', 'configs', 'requirements'))):
        raise ValueError('Choose a separate output directory, e.g. /workspace/pilot-runs/one')
    root.mkdir(parents=True, exist_ok=False)
    nav = replace(nav, output_dir=str(root))
    entries = [dict(index=i, source_scene_id=scene, source_episode_id=eid, status='pending',
                    success=False, error=None, trajectory=None, video=None)
               for i, (scene, eid) in enumerate(keys)]
    manifest = dict(schema_version=1, workflow='gibson_pilot', requested_episodes=len(entries),
                    complete=False, seed=nav.seed, effective_settings=effective_settings(sim, nav),
                    episodes=entries, visual_review='pending', real_data_acceptance='pending')
    write_manifest(root, manifest)
    scene_hashes = {}
    for definition, entry in zip(definitions, entries):
        entry['status'] = 'running'
        write_manifest(root, manifest)
        started = time.perf_counter()
        try:
            if definition['scene'] not in scene_hashes:
                scene_hashes[definition['scene']] = sha256_file(definition['scene'])
            episode = runner(definition, sim, nav)
            entry['simulation_seconds'] = time.perf_counter() - started
            episode['metadata'].update(episode_id=entry['index'],
                                       scene_sha256=scene_hashes[definition['scene']],
                                       simulation_seconds=entry['simulation_seconds'])
            validate_pilot(episode)
            filename = f"episode_{entry['index']:06d}.pt"
            tmp = root / (filename + '.tmp')
            torch.save(episode, tmp)
            tmp.replace(root / filename)
            entry.update(manifest_entry(entry['index'], episode, filename))
            entry['trajectory_bytes'] = (root / filename).stat().st_size
            # Commit the executed rollout even if encoding fails.
            entry['status'] = 'recorded'
            write_manifest(root, manifest)
            video = export_video(episode, root / f"episode_{entry['index']:06d}.mp4", nav.fps, composite=True)
            entry.update(status='finished', video=video.name, video_bytes=video.stat().st_size)
        except Exception as exc:
            entry['status'] = 'video_error' if entry['trajectory'] else 'error'
            entry['error'] = f'{type(exc).__name__}: {exc}'
        entry['total_seconds'] = time.perf_counter() - started
        write_manifest(root, manifest)
        print(f"scene={entry['source_scene_id']} episode={entry['source_episode_id']} "
              f"status={entry['status']} success={entry['success']} "
              f"termination={entry.get('termination')} error={entry['error']}", flush=True)
    manifest['complete'] = True  # means every planned attempt was accounted for
    manifest['summary'] = dict(attempted=len(entries), successes=sum(e['success'] for e in entries),
                               runtime_errors=sum(e['status'] == 'error' for e in entries),
                               video_errors=sum(e['status'] == 'video_error' for e in entries),
                               total_seconds=sum(e['total_seconds'] for e in entries),
                               total_artifact_bytes=sum(e.get('trajectory_bytes', 0) + e.get('video_bytes', 0) for e in entries))
    write_manifest(root, manifest)
    return manifest


def compare_episodes(first, second, atol=1e-6):
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    for episode in (first, second):
        validate_pilot(episode)
    a, b = first['metadata'], second['metadata']
    identity = ['source_scene_id', 'source_episode_id', 'source_episode_sha256',
                'source_definition', 'scene_sha256', 'seed', 'habitat_sim_version', 'habitat_lab_version',
                'forward_step_m', 'turn_degrees', 'goal_radius_m', 'packages', 'platform', 'controller']
    settings_a, settings_b = a['effective_settings'], b['effective_settings']
    # The output destination is intentionally different on a repeat run.
    nav_a = {k: v for k, v in settings_a['navigation'].items() if k != 'output_dir'}
    nav_b = {k: v for k, v in settings_b['navigation'].items() if k != 'output_dir'}
    if any(a[k] != b[k] for k in identity) or settings_a['simulator'] != settings_b['simulator'] or nav_a != nav_b:
        raise ValueError('Repeatability requires the same official episode, assets, versions, seed and settings')
    report = dict(atol=atol, fields={})
    for key in ('actions', 'positions', 'rotations_xyzw', 'headings', 'relative_goals', 'observations', 'collisions'):
        x, y = first[key], second[key]
        same_shape = x.shape == y.shape and x.dtype == y.dtype
        exact = same_shape and torch.equal(x, y)
        close = same_shape and (exact or (x.is_floating_point() and torch.allclose(x, y, atol=atol, rtol=0)))
        report['fields'][key] = dict(exact=bool(exact), within_tolerance=bool(close),
                                   max_abs_difference=float((x.double() - y.double()).abs().max()) if same_shape and x.numel() else None)
    report['metrics_equal'] = first['metrics'] == second['metrics']
    report['repeatable'] = report['metrics_equal'] and all(v['within_tolerance'] for v in report['fields'].values())
    return report


def validate_pilot_run(run_dir):
    """Re-read trajectories and decode videos, cross-checking the attempt ledger."""
    import imageio.v2 as imageio
    root = Path(run_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('workflow') != 'gibson_pilot' or manifest.get('schema_version') != 1:
        raise ValueError('Expected a Gibson pilot manifest')
    entries = manifest['episodes']
    if len(entries) != manifest['requested_episodes']:
        raise ValueError('Attempt ledger must include every planned episode')
    listed = set()
    for index, entry in enumerate(entries):
        if entry['index'] != index or entry['status'] not in ('pending', 'running', 'recorded', 'finished', 'error', 'video_error'):
            raise ValueError('Invalid attempt index/status')
        if manifest['complete'] and entry['status'] in ('pending', 'running', 'recorded'):
            raise ValueError('Complete manifest contains an unfinished attempt')
        if entry['status'] in ('error', 'video_error') and not entry['error']:
            raise ValueError('Failed attempt must explain its error')
        filename = entry['trajectory']
        if filename:
            if filename != f'episode_{index:06d}.pt':
                raise ValueError('Unexpected trajectory filename')
            episode = torch.load(root / filename, map_location='cpu', weights_only=True)
            validate_pilot(episode)
            expected = manifest_entry(index, episode, filename)
            if any(entry.get(k) != v for k, v in expected.items()):
                raise ValueError('Manifest does not match trajectory')
            if entry['trajectory_bytes'] != (root / filename).stat().st_size:
                raise ValueError('Trajectory storage measurement mismatch')
            listed.add(filename)
            if entry['video']:
                if entry['video'] != f'episode_{index:06d}.mp4':
                    raise ValueError('Unexpected video filename')
                path = root / entry['video']
                if path.stat().st_size != entry['video_bytes']:
                    raise ValueError('Video storage measurement mismatch')
                with imageio.get_reader(str(path)) as reader:
                    if reader.count_frames() != len(episode['actions']) + 1:
                        raise ValueError('Video must contain exactly T+1 frames')
                    expected_shape = tuple(episode['visualization_frames'].shape[1:])
                    if reader.get_data(0).shape != expected_shape:
                        raise ValueError('Decoded video dimensions differ from composite frames')
            elif entry['status'] == 'finished':
                raise ValueError('Finished attempt has no video')
        elif entry['status'] in ('finished', 'video_error', 'recorded'):
            raise ValueError('Recorded attempt has no trajectory')
    if listed != {p.name for p in root.glob('*.pt')}:
        raise ValueError('Unlisted trajectory files')
    return manifest

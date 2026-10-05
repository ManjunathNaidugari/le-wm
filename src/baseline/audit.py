"""Audit the actual pilot attempt ledger and raw RGB; heuristic flags need review."""
from pathlib import Path
import numpy as np
import torch
from jepa_navigation.data.gibson_pilot import validate_pilot
from jepa_navigation.data.rgb_qc import rgb_quality, THRESHOLDS
from .common import ACTIONS, building, contained_file, read_json, sha256_file, write_json


def audit_run(run_dir, output, exclusions=None, negligible_m=0.01):
    if not np.isfinite(negligible_m) or negligible_m < 0:
        raise ValueError('negligible_m must be finite and nonnegative')
    root = Path(run_dir).resolve()
    manifest = read_json(root / 'manifest.json')
    if manifest.get('workflow') not in ('gibson_pilot', 'navigation_collection'):
        raise ValueError('Expected pilot/collection attempt manifest')
    exclusions = exclusions or {}
    if any(not isinstance(reason, str) or not reason.strip() for reason in exclusions.values()):
        raise ValueError('Every explicit exclusion needs a reason')
    rows, seen = [], set()
    for entry in manifest['episodes']:
        key = f"{building(entry['source_scene_id'])}/{entry['source_episode_id']}"
        if key.casefold() in seen:
            raise ValueError(f'Duplicate official episode identity: {key}')
        seen.add(key.casefold())
        row = dict(identity=key, building=building(entry['source_scene_id']),
                   source_scene_id=entry['source_scene_id'], source_episode_id=str(entry['source_episode_id']),
                   status=entry['status'], success=entry.get('success', False),
                   trajectory=entry.get('trajectory'), structural_errors=[], review_flags=[],
                   included=False, inclusion_reason=None)
        filename = entry.get('trajectory')
        if not filename:
            row['structural_errors'].append(entry.get('error') or 'No recorded trajectory')
        else:
            try:
                path = contained_file(root, filename)
                row['trajectory'] = str(path)
                row['source_sha256'] = sha256_file(path)
                episode = torch.load(path, map_location='cpu', weights_only=True)
                try:
                    validate_pilot(episode)
                    if str(episode['metadata']['source_episode_id']) != row['source_episode_id'] or episode['metadata']['source_scene_id'] != row['source_scene_id']:
                        raise ValueError('Manifest/source identity mismatch')
                    if entry.get('steps') != episode['metrics']['steps'] or entry.get('success') != episode['metrics']['success']:
                        raise ValueError('Manifest/trajectory metrics mismatch')
                    if entry.get('trajectory_sha256') and entry['trajectory_sha256'] != row['source_sha256']:
                        raise ValueError('Collection trajectory hash mismatch')
                except Exception as exc:
                    row['structural_errors'].append(f'{type(exc).__name__}: {exc}')
                qc = rgb_quality(episode)
                row['review_flags'] = qc['flags']
                # Source observation indices, including terminal state T; not MP4 indices.
                flagged_frames = [f['frame'] for f in qc['frames'] if f.get('variance') is None or
                                  f['variance'] <= THRESHOLDS['low_variance'] or
                                  f['black_ratio'] >= THRESHOLDS['suspicious_pixel_ratio'] or
                                  f['saturation_ratio'] >= THRESHOLDS['suspicious_pixel_ratio']]
                row['review_frame_indices'] = flagged_frames
                row['rgb_summary'] = {k: qc[k] for k in ('mean_brightness', 'black_ratio', 'saturation_ratio', 'motion_frozen_ratio')}
                actions = episode['actions'].numpy()
                counts = np.bincount(actions, minlength=4)
                collisions = episode['collisions'].numpy()
                forward = actions == 0
                movement = actions != 3
                displacement = np.linalg.norm(np.diff(episode['positions'].numpy(), axis=0), axis=1)
                stuck = forward & (displacement <= negligible_m)
                row.update(steps=len(actions), action_counts=dict(zip(ACTIONS, counts.tolist())),
                           collision_count=int(collisions.sum()), movement_actions=int(movement.sum()),
                           collision_rate_per_movement_action=float(collisions.sum() / max(1, movement.sum())),
                           forward_actions=int(forward.sum()), negligible_forward_count=int(stuck.sum()),
                           negligible_forward_rate=float(stuck.sum() / max(1, forward.sum())),
                           negligible_forward_action_indices=np.flatnonzero(stuck).tolist(),
                           frozen_motion_action_indices=[int(i) for i in np.flatnonzero(movement)
                               if np.abs(episode['observations'][i+1].numpy().astype(float) -
                                         episode['observations'][i].numpy().astype(float)).mean() <= THRESHOLDS['frozen_mean_absolute_difference']],
                           effective_settings=episode['metadata']['effective_settings'])
                row['review_frame_indices'] = sorted(set(flagged_frames + [i + 1 for i in row['frozen_motion_action_indices']]))
            except Exception as exc:
                row['structural_errors'].append(f'{type(exc).__name__}: {exc}')
        if key in exclusions:
            row['inclusion_reason'] = 'explicit exclusion: ' + exclusions[key]
        elif row['structural_errors']:
            row['inclusion_reason'] = 'structural failure; repair/review required'
        else:
            row['included'] = True
            row['inclusion_reason'] = 'included; visual flags remain for review'
        rows.append(row)
    unknown = set(exclusions) - {r['identity'] for r in rows}
    if unknown:
        raise ValueError(f'Exclusions refer to unknown identities: {sorted(unknown)}')
    report = dict(schema_version=1, workflow='baseline_audit', run_dir=str(root),
                  manifest_sha256=sha256_file(root / 'manifest.json'),
                  collision_rate_denominator='non-STOP actions (FORWARD/LEFT/RIGHT); zero denominator yields 0',
                  forward_rate_denominator='FORWARD actions; zero denominator yields 0',
                  negligible_displacement_threshold_m=negligible_m,
                  note='Heuristics flag review; they do not prove corruption. No data is deleted.',
                  rgb_thresholds=THRESHOLDS, episodes=rows,
                  summary=dict(attempted=len(rows), included=sum(r['included'] for r in rows),
                               structural_errors=sum(bool(r['structural_errors']) for r in rows),
                               review_flagged=sum(bool(r['review_flags']) for r in rows)))
    write_json(output, report, overwrite=False)
    return report


def combine_audits(inputs, output):
    """Combine train/development reports without dropping exclusions or provenance."""
    reports = [read_json(path) for path in inputs]
    if not reports or any(r.get('workflow') != 'baseline_audit' for r in reports):
        raise ValueError('Expected one or more baseline audit reports')
    settings = ('collision_rate_denominator', 'forward_rate_denominator',
                'negligible_displacement_threshold_m', 'rgb_thresholds')
    if any(any(r[k] != reports[0][k] for k in settings) for r in reports):
        raise ValueError('Cannot combine audits with different thresholds/denominators')
    rows = [row for report in reports for row in report['episodes']]
    keys = [r['identity'].casefold() for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate official episode identities across audits')
    result = dict(schema_version=1, workflow='baseline_audit',
                  source_audits=[dict(path=str(Path(p).resolve()), sha256=sha256_file(p)) for p in inputs],
                  **{k: reports[0][k] for k in settings},
                  note='Combined inclusion/exclusion records; no data deleted.', episodes=rows,
                  summary=dict(attempted=len(rows), included=sum(r['included'] for r in rows),
                               structural_errors=sum(bool(r['structural_errors']) for r in rows),
                               review_flagged=sum(bool(r['review_flags']) for r in rows)))
    write_json(output, result, overwrite=False)
    return result

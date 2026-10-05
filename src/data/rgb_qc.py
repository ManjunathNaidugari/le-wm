"""RGB data-quality heuristics, not navigation or scientific evaluation metrics."""
import json
from pathlib import Path
import numpy as np
import torch

THRESHOLDS = dict(black_pixel_max=5, saturated_channel_min=250,
                  suspicious_pixel_ratio=0.8, low_variance=4.0,
                  frozen_mean_absolute_difference=0.5, frozen_ratio=0.8,
                  minimum_motion_pairs=3)


def rgb_quality(trajectory):
    rgb = trajectory.get('observations')
    result = dict(image_shape=list(rgb.shape) if isinstance(rgb, torch.Tensor) else None,
                  nonfinite_values=0, mean_brightness=None, black_ratio=None,
                  saturation_ratio=None, frozen_ratio=None, motion_frozen_ratio=None,
                  min_y=None, max_y=None, frames=[], flags=[])
    flags = result['flags']
    positions = trajectory.get('positions')
    if isinstance(positions, torch.Tensor) and positions.ndim == 2 and positions.shape[1] == 3 and len(positions):
        y = positions[:, 1].detach().cpu().numpy()
        if np.isfinite(y).all():
            result['min_y'], result['max_y'] = float(y.min()), float(y.max())
        else:
            flags.append('nonfinite_y')
    else:
        flags.append('invalid_positions')
    if not isinstance(rgb, torch.Tensor) or rgb.ndim != 4 or rgb.shape[1] != 3 or min(rgb.shape) < 1:
        flags.append('invalid_rgb_dimensions')
        result['qc_status'] = 'flagged'
        return result
    if rgb.dtype != torch.uint8:
        flags.append('invalid_rgb_dtype_expected_uint8')
    previous = None
    frozen, motion_frozen = [], []
    actions = trajectory.get('actions')
    for index, tensor in enumerate(rgb):
        frame = tensor.detach().cpu().numpy().astype(np.float32)
        finite = np.isfinite(frame)
        missing = int((~finite).sum())
        result['nonfinite_values'] += missing
        if missing:
            result['frames'].append(dict(frame=index, mean_brightness=None, variance=None,
                                         black_ratio=None, saturation_ratio=None))
            previous = None
            continue
        if (frame < 0).any() or (frame > 255).any():
            if 'out_of_range_rgb' not in flags:
                flags.append('out_of_range_rgb')
        stats = dict(frame=index, mean_brightness=float(frame.mean()), variance=float(frame.var()),
                     black_ratio=float((frame.max(axis=0) <= THRESHOLDS['black_pixel_max']).mean()),
                     saturation_ratio=float((frame.max(axis=0) >= THRESHOLDS['saturated_channel_min']).mean()))
        result['frames'].append(stats)
        if previous is not None:
            unchanged = bool(np.abs(frame - previous).mean() <= THRESHOLDS['frozen_mean_absolute_difference'])
            frozen.append(unchanged)
            if isinstance(actions, torch.Tensor) and actions.ndim == 1 and index-1 < len(actions) and actions[index-1].item() != 3:
                motion_frozen.append(unchanged)
        previous = frame
    if result['nonfinite_values']:
        flags.append('nonfinite_rgb')
    valid = [f for f in result['frames'] if f['variance'] is not None]
    if valid:
        for field in ('mean_brightness', 'black_ratio', 'saturation_ratio'):
            result[field] = float(np.mean([f[field] for f in valid]))
        # Flag even one severely clipped frame; report counts for triage.
        for field, label in [('black_ratio', 'near_black_frames'), ('saturation_ratio', 'saturated_frames')]:
            count = sum(f[field] >= THRESHOLDS['suspicious_pixel_ratio'] for f in valid)
            result[label] = count
            if count:
                flags.append(label)
        count = sum(f['variance'] <= THRESHOLDS['low_variance'] for f in valid)
        result['low_variance_frames'] = count
        if count:
            flags.append('low_variance_frames')
    result['frozen_ratio'] = float(np.mean(frozen)) if frozen else 0.
    result['motion_frozen_ratio'] = float(np.mean(motion_frozen)) if motion_frozen else 0.
    result['motion_pairs'] = len(motion_frozen)
    if len(motion_frozen) >= THRESHOLDS['minimum_motion_pairs'] and result['motion_frozen_ratio'] >= THRESHOLDS['frozen_ratio']:
        flags.append('frozen_motion_frames')
    result['qc_status'] = 'flagged' if flags else 'pass'
    return result


def qc_dataset(dataset_dir):
    """Inspect every .pt, including structurally invalid episodes; never mutate data."""
    from jepa_navigation.data.habitat_dataset import validate_trajectory
    root = Path(dataset_dir)
    paths = sorted(root.glob('*.pt'))
    if not paths:
        raise ValueError(f'No trajectories in {root}')
    rows = []
    for path in paths:
        row = dict(trajectory=path.name, episode=path.stem, scene=None, steps=None)
        try:
            trajectory = torch.load(path, map_location='cpu', weights_only=True)
            row.update(rgb_quality(trajectory))
            meta = trajectory.get('metadata', {})
            row.update(episode=meta.get('episode_id', path.stem), scene=meta.get('scene'),
                       steps=trajectory.get('metrics', {}).get('steps'))
            try:
                validate_trajectory(trajectory, check_rgb_quality=False)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                row['flags'].append('invalid_trajectory')
                row['validation_error'] = str(exc)
                row['qc_status'] = 'flagged'
        except Exception as exc:
            row.update(qc_status='flagged', flags=['unreadable_trajectory'], validation_error=str(exc))
        rows.append(row)
    report = dict(thresholds=THRESHOLDS, note='DATA QUALITY HEURISTICS; review flags, do not treat as scientific metrics.',
                  total=len(rows), flagged=sum(r['qc_status'] == 'flagged' for r in rows), episodes=rows)
    tmp = root / 'qc_report.json.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(root / 'qc_report.json')
    return report

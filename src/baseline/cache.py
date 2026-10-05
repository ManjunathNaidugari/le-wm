"""Hash-bound resumable episode/chunk caches; one raw episode in memory at a time."""
from pathlib import Path
import numpy as np
import torch
from .features import raw_clip, FeatureConfig
from .common import artifact_lock, contained_file, fingerprint, read_json, sha256_file, write_json
from jepa_navigation.data.gibson_pilot import validate_pilot
from .provenance import preflight_audit, require_expert


def save_array(path, array):
    path = Path(path)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as stream:
        np.save(stream, np.asarray(array), allow_pickle=False)
    tmp.replace(path)


def extract_cache(audit_path, cache_dir, encoder, chunk_size=16):
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    selected = preflight_audit(audit_path)
    config = FeatureConfig(**encoder.identity['feature_config'])
    signature = dict(schema_version=2, encoder=encoder.identity,
                     audit_sha256=sha256_file(audit_path), chunk_size=chunk_size,
                     sources=[dict(identity=r['identity'], path=r['trajectory'], sha256=r['source_sha256'],
                                   provenance=r['provenance']) for r in selected])
    root = Path(cache_dir).resolve()
    with artifact_lock(root):
        manifest_path = root / 'cache.json'
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest['signature'] != signature:
                raise ValueError('Stale/incompatible feature cache; choose a new directory')
        else:
            if any(p.name != '.writer.lock' for p in root.iterdir()):
                raise FileExistsError('Refusing to adopt an unrelated nonempty cache directory')
            manifest = dict(workflow='causal_feature_cache', signature=signature, fingerprint=fingerprint(signature), complete=False, episodes=[])
            write_json(manifest_path, manifest)
        for index, row in enumerate(selected):
            if sha256_file(row['trajectory']) != row['source_sha256']:
                raise ValueError(f"Source changed since audit: {row['identity']}")
            episode = torch.load(row['trajectory'], map_location='cpu', weights_only=True)
            validate_pilot(episode)
            require_expert(episode, row['provenance'])
            # Never select video/composite/map tensors as encoder inputs.
            observations = episode['observations']
            t = len(episode['actions'])
            if t < 1:
                raise ValueError('Zero-action episode cannot provide behavior cloning targets')
            if index < len(manifest['episodes']):
                item = manifest['episodes'][index]
                if item['identity'] != row['identity'] or item['steps'] != t:
                    raise ValueError('Cache source/label alignment changed')
            else:
                item = dict(identity=row['identity'], building=row['building'], steps=t,
                            source_path=row['trajectory'], source_sha256=row['source_sha256'], chunks=[], provenance=row['provenance'],
                            action_names=episode['action_names'], effective_settings=row['effective_settings'],
                            targets=f'targets_{index:06d}.npz')
                manifest['episodes'].append(item)
            targets = root / item['targets']
            if not targets.exists():
                tmp = targets.with_name(targets.name + '.partial')
                with tmp.open('wb') as stream:
                    np.savez(stream, actions=episode['actions'].numpy(), goals=episode['relative_goals'][:-1].numpy())
                tmp.replace(targets)
                item['targets_sha256'] = sha256_file(targets)
            elif 'targets_sha256' not in item:
                # Recover a targets file committed immediately before a process interruption.
                with np.load(targets, allow_pickle=False) as saved:
                    if not np.array_equal(saved['actions'], episode['actions'].numpy()) or not np.array_equal(saved['goals'], episode['relative_goals'][:-1].numpy()):
                        raise ValueError('Uncommitted cache targets differ from the source')
                item['targets_sha256'] = sha256_file(targets)
            elif sha256_file(targets) != item['targets_sha256']:
                raise ValueError('Cache targets are stale/corrupt')
            write_json(manifest_path, manifest)
            for start in range(0, t, chunk_size):
                end = min(t, start + chunk_size)
                existing = [c for c in item['chunks'] if c['start'] == start]
                if existing:
                    chunk = existing[0]
                    if chunk['end'] != end or sha256_file(root / chunk['file']) != chunk['sha256']:
                        raise ValueError('Committed cache chunk is corrupt/incompatible')
                    continue
                # At most one chunk of pooled features plus one clip is retained.
                values = np.stack([encoder.encode(raw_clip(observations, action_t, config)) for action_t in range(start, end)])
                expected = (end - start, 2 * config.grid_size**2, encoder.identity['feature_dimension'])
                if values.shape != expected or values.dtype != np.float32 or not np.isfinite(values).all():
                    raise ValueError('Encoder feature shape/dtype is incompatible')
                path = root / f'features_{index:06d}_{start:06d}.npy'
                # An uncommitted orphan chunk can be replaced: it was never advertised as valid.
                save_array(path, values)
                item['chunks'].append(dict(start=start, end=end, file=path.name, sha256=sha256_file(path)))
                write_json(manifest_path, manifest)
            del episode, observations
        manifest['complete'] = True
        write_json(manifest_path, manifest)
        return manifest


def validate_cache(root, encoder_identity=None, verify_sources=True):
    root = Path(root)
    manifest = read_json(root / 'cache.json')
    if manifest.get('workflow') != 'causal_feature_cache' or not manifest.get('complete'):
        raise ValueError('Feature extraction is incomplete or has an unknown schema')
    signature = manifest['signature']
    if fingerprint(signature) != manifest['fingerprint'] or signature['schema_version'] != 2:
        raise ValueError('Cache signature is invalid')
    if encoder_identity is not None and encoder_identity != signature['encoder']:
        raise ValueError('Cache encoder/preprocessing/history/pooling mismatch')
    config = FeatureConfig(**signature['encoder']['feature_config'])
    if len(manifest['episodes']) != len(signature['sources']):
        raise ValueError('Cache episode count mismatch')
    for item, source in zip(manifest['episodes'], signature['sources']):
        if item['identity'] != source['identity'] or item['source_path'] != source['path'] or item['source_sha256'] != source['sha256']:
            raise ValueError('Cache source identity mismatch')
        if verify_sources and sha256_file(item['source_path']) != item['source_sha256']:
            raise ValueError('Cache is stale relative to raw trajectories')
        if item.get('provenance') != source.get('provenance') or not item.get('provenance'):
            raise ValueError('Missing/inconsistent expert provenance; re-audit and rebuild cache')
        if verify_sources:
            episode = torch.load(item['source_path'], map_location='cpu', weights_only=True)
            validate_pilot(episode)
            require_expert(episode, item['provenance'])
        target_path = contained_file(root, item['targets'])
        if sha256_file(target_path) != item['targets_sha256']:
            raise ValueError('Cache targets are corrupt')
        with np.load(target_path, allow_pickle=False) as targets:
            if targets['actions'].shape != (item['steps'],) or targets['actions'].dtype != np.int64 or targets['goals'].shape != (item['steps'], 3) or not np.isfinite(targets['goals']).all():
                raise ValueError('Feature/action/goal alignment mismatch')
            if np.any((targets['actions'] < 0) | (targets['actions'] > 3)):
                raise ValueError('Invalid cached actions')
        next_start = 0
        for chunk in sorted(item['chunks'], key=lambda c: c['start']):
            if chunk['start'] != next_start or chunk['end'] <= next_start:
                raise ValueError('Cache chunks must cover consecutive action timesteps')
            path = contained_file(root, chunk['file'])
            if sha256_file(path) != chunk['sha256']:
                raise ValueError('Corrupt feature chunk')
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if values.shape != (chunk['end']-chunk['start'], 2*config.grid_size**2, signature['encoder']['feature_dimension']) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError('Invalid feature chunk shape/dtype/values')
            next_start = chunk['end']
        if next_start != item['steps'] or item['action_names'] != ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP']:
            raise ValueError('Feature/action contract mismatch')
    return manifest

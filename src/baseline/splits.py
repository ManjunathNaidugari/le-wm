"""Explicit building-disjoint manifests; inventory reflects files actually present."""
from pathlib import Path
from jepa_navigation.data.pointnav import _read, resolve_scene, load_pointnav
from .common import PILOT_BUILDINGS, building, identity, fingerprint, read_json, sha256_file, write_json

PHASES = ('train', 'development', 'final_evaluation')


def inspect_availability(episode_data, scene_data_dir, output):
    root = Path(scene_data_dir).resolve()
    available, errors = {}, []
    for index_path in episode_data:
        index = Path(index_path).resolve()
        try:
            header = _read(index)
            index_digest = sha256_file(index)
            template = header.get('content_scenes_path', '{data_path}/content/{scene}.json.gz')
            # The official default path only; custom templates are handled by the loader after selection.
            sources = [index] + sorted((index.parent / 'content').glob('*.json.gz'))
            if template != '{data_path}/content/{scene}.json.gz':
                raise ValueError('Inventory currently requires the official default content shard layout')
            seen = set()
            for source in sources:
                try:
                    doc = header if source == index else _read(source)
                    for raw in doc['episodes']:
                        scene = raw['scene_id']
                        mesh = resolve_scene(scene, root, require_exists=False)
                        key = (scene, str(raw['episode_id']))
                        if key in seen:
                            raise ValueError(f'Duplicate episode in official input: {key}')
                        seen.add(key)
                        name = building(scene)
                        item = available.setdefault(name, dict(building=name, scene_id=scene, mesh=str(mesh),
                                                                mesh_exists=mesh.is_file(), inputs=[]))
                        if item['scene_id'] != scene:
                            raise ValueError(f'Ambiguous building paths: {name}')
                        inputs = [i for i in item['inputs'] if i['episode_data'] == str(index)]
                        if not inputs:
                            item['inputs'].append(dict(episode_data=str(index), index_sha256=index_digest, count=0))
                        next(i for i in item['inputs'] if i['episode_data'] == str(index))['count'] += 1
                except Exception as exc:
                    errors.append(dict(file=str(source), error=f'{type(exc).__name__}: {exc}'))
        except Exception as exc:
            errors.append(dict(file=str(index), error=f'{type(exc).__name__}: {exc}'))
    report = dict(schema_version=1, workflow='official_asset_intersections', scene_data_dir=str(root),
                  buildings=[available[k] for k in sorted(available)], errors=errors,
                  note='Only existing meshes intersecting official episode definitions are collection candidates; no downloads.')
    write_json(output, report, overwrite=False)
    return report


def validate_splits(plan):
    if plan.get('workflow') != 'building_disjoint_navigation' or plan.get('schema_version') != 1:
        raise ValueError('Unknown split schema')
    payload = {k: v for k, v in plan.items() if k != 'fingerprint'}
    if plan.get('fingerprint') != fingerprint(payload):
        raise ValueError('Split manifest fingerprint mismatch')
    buildings_seen, episodes_seen = set(), set()
    for phase in PHASES:
        part = plan['splits'][phase]
        names = [n.casefold() for n in part['buildings']]
        if len(set(names)) != len(names) or buildings_seen.intersection(names):
            raise ValueError('Buildings overlap between train/development/final evaluation')
        buildings_seen.update(names)
        present = set()
        for episode in part['episodes']:
            key = tuple(identity(episode))
            if key in episodes_seen:
                raise ValueError('Duplicate episode identity in saved split manifests')
            episodes_seen.add(key)
            present.add(key[0])
            if key[0] not in names:
                raise ValueError('Episode is not assigned to one of its split buildings')
        if present != set(names):
            raise ValueError('Every named building must have materialized episodes')
    final_names = {n.casefold() for n in plan['splits']['final_evaluation']['buildings']}
    if final_names.intersection(n.casefold() for n in plan['pilot_buildings'] + PILOT_BUILDINGS):
        raise ValueError('Pilot buildings cannot be used for final evaluation')
    if plan['purpose'] not in ('integration', 'experiment'):
        raise ValueError('Unknown split purpose')
    if not plan['splits']['train']['episodes']:
        raise ValueError('Training split has no included episodes')
    if plan['purpose'] == 'experiment' and (not plan['splits']['train']['episodes'] or not plan['splits']['development']['episodes'] or not plan['splits']['final_evaluation']['episodes']):
        raise ValueError('Experiment needs explicit train, development and final-evaluation episodes')
    excluded = set()
    permitted = {tuple(identity(e)) for phase in ('train', 'development') for e in plan['splits'][phase]['episodes']}
    for item in plan.get('training_exclusions', []):
        key = tuple(item['identity'])
        if key not in permitted or key in excluded or not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('Invalid/duplicate training exclusion; evaluation identities must stay intact')
        excluded.add(key)
    return plan


def training_definitions(plan, phase):
    excluded = {tuple(item['identity']) for item in plan.get('training_exclusions', [])}
    return [e for e in plan['splits'][phase]['episodes'] if tuple(identity(e)) not in excluded]


def resolve_training_inputs(split_path, audit_path, output, reason):
    """New immutable manifest: explicit label exclusions, unchanged evaluation requests."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('A documented dataset-resolution reason is required')
    plan = validate_splits(read_json(split_path))
    audit = read_json(audit_path)
    if audit.get('schema_version') != 2 or audit.get('workflow') != 'baseline_audit':
        raise ValueError('Re-audit expert provenance first')
    rows = {(r['building'].casefold(), r['source_episode_id']): r for r in audit['episodes']}
    exclusions = {tuple(e['identity']): e for e in plan.get('training_exclusions', [])}
    for phase in ('train', 'development'):
        for definition in plan['splits'][phase]['episodes']:
            key = tuple(identity(definition))
            if key not in rows:
                raise ValueError(f'Cannot resolve an unaudited requested identity: {key}')
            if not rows[key]['included']:
                exclusions[key] = dict(identity=list(key), reason=reason + ': ' + rows[key]['inclusion_reason'])
    parent = plan['fingerprint']
    plan.update(training_exclusions=list(exclusions.values()), parent_split_fingerprint=parent,
                training_resolution=dict(reason=reason, audit_sha256=sha256_file(audit_path)),
                evaluation_requests_unchanged=True)
    plan.pop('fingerprint')
    plan['fingerprint'] = fingerprint(plan)
    validate_splits(plan)
    write_json(output, plan, overwrite=False)
    return plan


def prepare_splits(inventory_path, scene_groups, counts, output, seed=42, pilot_buildings=None):
    inventory = read_json(inventory_path)
    if inventory.get('workflow') != 'official_asset_intersections' or inventory['errors']:
        raise ValueError('Inspect and resolve availability inventory errors before materializing splits')
    available = {r['building']: r for r in inventory['buildings']}
    pilot_buildings = list(pilot_buildings or PILOT_BUILDINGS)
    splits = {}
    for phase in PHASES:
        names = list(scene_groups[phase])
        count = counts[phase]
        if not names or len(set(names)) != len(names) or type(count) is not int or count < len(names):
            raise ValueError('Choose explicit unique buildings and enough episodes for each split')
        episodes = []
        for i, name in enumerate(names):
            if name not in available or not available[name]['mesh_exists']:
                raise ValueError(f'Building has no locally available official mesh/episode intersection: {name}')
            item = available[name]
            if len(item['inputs']) != 1:
                raise ValueError(f'Choose an unambiguous official input for {name}')
            source = item['inputs'][0]
            if sha256_file(source['episode_data']) != source['index_sha256']:
                raise ValueError('Official index changed since inventory')
            requested = count // len(names) + int(i < count % len(names))
            if source['count'] < requested:
                raise ValueError(f'{name} has fewer than {requested} available official episodes')
            selected = load_pointnav(source['episode_data'], inventory['scene_data_dir'], requested,
                                     seed=seed, scenes=[name])
            scene_hash = sha256_file(selected[0]['scene'])
            for definition in selected:
                definition['scene_sha256'] = scene_hash
            episodes.extend(selected)
        splits[phase] = dict(buildings=names, episodes=episodes)
    plan = dict(schema_version=1, workflow='building_disjoint_navigation', purpose='experiment',
                seed=seed, pilot_buildings=pilot_buildings, inventory_sha256=sha256_file(inventory_path), splits=splits)
    plan['fingerprint'] = fingerprint(plan)
    validate_splits(plan)
    write_json(output, plan, overwrite=False)
    return plan


def integration_split(audit_path, output):
    audit = read_json(audit_path)
    if audit.get('workflow') != 'baseline_audit':
        raise ValueError('Expected audited pilot inclusion records')
    episodes = []
    import torch
    for row in audit['episodes']:
        if not row['included']:
            raise ValueError('Integration split would omit requested recordings; repair inputs or resolve training exclusions on an existing split explicitly')
        if row['included']:
            if sha256_file(row['trajectory']) != row['source_sha256']:
                raise ValueError('Pilot source changed after audit')
            trajectory = torch.load(row['trajectory'], map_location='cpu', weights_only=True)
            meta = trajectory['metadata']
            episodes.append({k: meta[k] for k in ('source_scene_id', 'source_episode_id', 'source_episode_sha256',
                                                 'source_definition', 'scene', 'start_position', 'start_rotation_xyzw',
                                                 'goal_position', 'source_goal_radius', 'source_episode_file')})
            episodes[-1]['scene_sha256'] = meta['scene_sha256']
    names = sorted({building(e['source_scene_id']) for e in episodes})
    plan = dict(schema_version=1, workflow='building_disjoint_navigation', purpose='integration', seed=42,
                pilot_buildings=sorted(set(PILOT_BUILDINGS + names)), audit_sha256=sha256_file(audit_path),
                splits={p: dict(buildings=names if p == 'train' else [], episodes=episodes if p == 'train' else []) for p in PHASES})
    plan['fingerprint'] = fingerprint(plan)
    validate_splits(plan)
    write_json(output, plan, overwrite=False)
    return plan

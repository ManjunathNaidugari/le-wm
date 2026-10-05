"""Expert-label eligibility is independent of structural recording validity."""
from .common import read_json, sha256_file

EXPERT_CONTROLLER = 'ShortestPathFollower(stop_on_error=False)'
POLICY_CONTROLLERS = {'supplied_policy', 'goal_only_policy', 'frozen_vjepa2_direct_policy'}


def classify_provenance(context, metadata, resolution=None):
    controller = metadata.get('controller')
    purpose = context.get('purpose')
    learned = (controller in POLICY_CONTROLLERS or 'policy_encoder' in metadata or
               purpose in ('learned_evaluation', 'integration_rollout') or
               (context.get('workflow') == 'navigation_collection' and
                context.get('controller') not in (None, 'expert')))
    synthetic = bool(metadata.get('synthetic', False) or
                     any(str(metadata.get(k, '')).startswith('unit-fixture')
                         for k in ('habitat_sim_version', 'habitat_lab_version')))
    resolution = resolution or None
    if synthetic:
        eligible, reason = False, 'synthetic recording cannot provide training labels'
    elif learned:
        eligible, reason = False, 'learned-policy recording cannot provide expert labels'
    elif controller == EXPERT_CONTROLLER and (
            context.get('workflow') == 'gibson_pilot' or
            (purpose == 'collection' and context.get('controller') == 'expert')):
        eligible, reason = True, 'verified recorded shortest-path expert controller'
    elif resolution is not None:
        if (not isinstance(resolution, dict) or resolution.get('kind') != 'expert' or
                any(not isinstance(resolution.get(k), str) or not resolution[k].strip()
                    for k in ('reason', 'evidence'))):
            raise ValueError('Expert resolution requires kind=expert, a reason and evidence reference')
        eligible, reason = True, 'explicit researcher resolution of unknown expert provenance'
    else:
        eligible, reason = False, 'unknown expert provenance; documented resolution required'
    return dict(context=context, recorded_controller=controller,
                recorded_workflow=metadata.get('workflow'), synthetic=synthetic,
                resolution=resolution, expert_eligible=eligible, reason=reason)


def require_expert(episode, provenance):
    if not isinstance(provenance, dict):
        raise ValueError('Missing expert provenance; re-audit and rebuild the cache')
    actual = classify_provenance(provenance['context'], episode['metadata'], provenance.get('resolution'))
    if actual != provenance or not actual['expert_eligible']:
        raise ValueError('Expert provenance mismatch or ineligible learned/unknown labels')
    if len(episode['actions']) == 0:
        raise ValueError('Zero-transition recording has no behavior-cloning labels')


def preflight_audit(path, split_path=None):
    """Validate *all* selected raw inputs before starting an expensive worker."""
    import torch
    from jepa_navigation.data.gibson_pilot import validate_pilot
    from .splits import validate_splits, training_definitions
    audit = read_json(path)
    if audit.get('workflow') != 'baseline_audit' or audit.get('schema_version') != 2:
        raise ValueError('Re-audit inputs with expert provenance (audit schema 2)')
    selected = [r for r in audit['episodes'] if r['included']]
    errors = []
    for row in selected:
        try:
            if not row.get('training_eligible') or not row.get('recording_valid'):
                raise ValueError('Audit selection contains an ineligible recording')
            if sha256_file(row['trajectory']) != row['source_sha256']:
                raise ValueError('Source changed after audit')
            episode = torch.load(row['trajectory'], map_location='cpu', weights_only=True)
            validate_pilot(episode)
            require_expert(episode, row.get('provenance'))
        except Exception as exc:
            errors.append(f"{row['identity']}: {exc}")
    if split_path:
        from .common import identity
        plan = validate_splits(read_json(split_path))
        required = {tuple(identity(e)) for phase in ('train', 'development')
                    for e in training_definitions(plan, phase)}
        available = {(r['building'].casefold(), r['source_episode_id']) for r in selected}
        errors.extend(f'{b}/{i}: missing/ineligible requested expert input; resolve the saved training exclusions explicitly'
                      for b, i in sorted(required - available))
    if not selected:
        errors.append('No eligible expert transitions; inspect inclusion reasons in the audit')
    if errors:
        raise ValueError('Training-input preflight failed:\n' + '\n'.join(errors))
    return selected

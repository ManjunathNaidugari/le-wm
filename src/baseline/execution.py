"""Resumable expert collection and restricted closed-loop policy execution."""
from dataclasses import replace, asdict
from pathlib import Path
import time
import numpy as np
import torch
from .common import artifact_lock, building, fingerprint, identity, read_json, sha256_file, write_json
from .features import CausalHistory, FeatureConfig
from .training import load_policy
from .splits import validate_splits
from jepa_navigation.data.gibson_pilot import run_pilot_episode, validate_pilot
from jepa_navigation.data.habitat_dataset_collection import manifest_entry
from jepa_navigation.utils.config import SimulatorConfig, NavigationConfig, effective_settings
from jepa_navigation.utils.video import export_video
from jepa_navigation.navigation.actions import Action


class OnlinePolicy:
    """Receives only RGB + Cartesian relative position; no expert/map/path API."""
    def __init__(self, model, metadata, encoder=None):
        self.model, self.metadata, self.encoder = model, metadata, encoder
        if not model.goal_only and (encoder is None or encoder.identity != metadata['encoder']):
            raise ValueError('Online encoder differs from training/cache identity')
        self.history = CausalHistory(FeatureConfig(**metadata['encoder']['feature_config']))
        self.latencies_ms = []

    def __call__(self, observation):
        if set(observation) != {'rgb', 'relative_goal'}:
            raise ValueError('Policy observation must contain only RGB and relative_goal')
        started = time.perf_counter()
        goals = torch.as_tensor(np.array(observation['relative_goal'], dtype=np.float32, copy=True)).reshape(1, 3)
        if not torch.isfinite(goals).all():
            raise ValueError('Nonfinite relative goal')
        if self.model.goal_only:
            features = None
        else:
            clip = self.history.append(observation['rgb'])
            features = torch.from_numpy(self.encoder.encode(clip)).unsqueeze(0)
        with torch.inference_mode():
            action = Action(int(self.model(features, goals).argmax(1).item()))
        self.latencies_ms.append((time.perf_counter() - started) * 1000)
        return action


def learned_runner(model, metadata, encoder=None):
    def run(definition, sim, nav):
        policy = OnlinePolicy(model, metadata, encoder)
        # The recorder retains poses/maps for metrics/videos; the policy never receives them.
        def restricted(state):
            return policy({'rgb': state['rgb'], 'relative_goal': state['relative_goal']})
        episode = run_pilot_episode(definition, sim, nav, controller=restricted)
        episode['metadata']['controller'] = 'goal_only_policy' if model.goal_only else 'frozen_vjepa2_direct_policy'
        episode['metadata']['policy_encoder'] = metadata['encoder']
        episode['metadata']['integration_only'] = metadata['integration_only']
        episode['inference_latency_ms'] = torch.tensor(policy.latencies_ms, dtype=torch.float64)
        episode['metrics']['mean_inference_latency_ms'] = float(np.mean(policy.latencies_ms)) if policy.latencies_ms else 0.
        episode['metrics']['max_inference_latency_ms'] = max(policy.latencies_ms, default=0.)
        return episode
    return run


def execute_plan(plan_path, phase, output_dir, sim, nav, runner=None, controller_identity='expert',
                 resume=False, retry_errors=False, videos=False, purpose='collection'):
    plan = validate_splits(read_json(plan_path))
    definitions = plan['splits'][phase]['episodes']
    if not definitions:
        raise ValueError('Chosen split has no requested episodes')
    runner = runner or run_pilot_episode
    root = Path(output_dir).resolve()
    source_root = Path(__file__).resolve().parents[2]
    if root == source_root or any(folder == root or folder in root.parents for folder in
                                 (source_root / x for x in ('src', 'scripts', 'tests', 'docs', 'configs', 'requirements', '.git'))):
        raise ValueError('Outputs must be separate from source/configuration directories')
    if root.exists() and not resume:
        raise FileExistsError('Existing collection/evaluation directory requires explicit --resume')
    nav = replace(nav, output_dir=str(root))
    signature = dict(plan_fingerprint=plan['fingerprint'], phase=phase, controller=controller_identity,
                     settings=effective_settings(sim, nav), videos=videos, purpose=purpose)
    with artifact_lock(root):
        path = root / 'manifest.json'
        if path.exists():
            manifest = read_json(path)
            if manifest['signature'] != signature:
                raise ValueError('Cannot resume with a different split/controller/settings')
            if manifest.get('requested_episodes') != len(definitions) or len(manifest['episodes']) != len(definitions):
                raise ValueError('Resume ledger does not account for every requested episode')
            for index, (entry, definition) in enumerate(zip(manifest['episodes'], definitions)):
                if entry['index'] != index or entry['definition'] != definition or identity(entry) != identity(definition):
                    raise ValueError('Resume ledger source definitions/order changed')
        else:
            if any(p.name != '.writer.lock' for p in root.iterdir()):
                raise FileExistsError('Cannot adopt unrelated existing outputs')
            manifest = dict(schema_version=1, workflow='navigation_collection', signature=signature,
                            requested_episodes=len(definitions), complete=False,
                            episodes=[dict(index=i, source_scene_id=d['source_scene_id'],
                                           source_episode_id=d['source_episode_id'], definition=d, status='pending',
                                           success=False, error=None, trajectory=None, video=None, attempts=[])
                                      for i, d in enumerate(definitions)])
            write_json(path, manifest)
        for entry, definition in zip(manifest['episodes'], definitions):
            if entry['status'] == 'finished':
                trajectory_path = root / entry['trajectory']
                if sha256_file(trajectory_path) != entry['trajectory_sha256']:
                    raise ValueError('Completed rollout changed on disk; refusing resume')
                validate_pilot(torch.load(trajectory_path, map_location='cpu', weights_only=True))
                if videos and (sha256_file(root / entry['video']) != entry['video_sha256']):
                    raise ValueError('Completed video changed/missing; refusing resume')
                continue
            if entry['status'] in ('error', 'video_error') and not retry_errors:
                continue
            if entry['attempts'] and entry['attempts'][-1]['status'] == 'running':
                entry['attempts'][-1].update(status='interrupted', error='Previous process interrupted; outputs not committed')
            attempt_id = len(entry['attempts'])
            attempt = dict(attempt=attempt_id, status='running', error=None)
            entry['attempts'].append(attempt)
            entry.update(status='running', error=None, trajectory=None, video=None, success=False)
            manifest['complete'] = False
            write_json(path, manifest)
            started = time.perf_counter()
            failure_stage = 'error'
            try:
                scene_hash = sha256_file(definition['scene'])
                if definition.get('scene_sha256') and scene_hash != definition['scene_sha256']:
                    raise ValueError('Scene changed after split materialization')
                if sha256_file(definition['source_episode_file']) != definition['source_episode_sha256']:
                    raise ValueError('Official episode shard changed after split materialization')
                episode = runner(definition, sim, nav)
                episode['metadata'].update(episode_id=entry['index'], scene_sha256=scene_hash)
                validate_pilot(episode)
                filename = f"episode_{entry['index']:06d}_attempt_{attempt_id:03d}.pt"
                target = root / filename
                if target.exists():
                    raise FileExistsError(target)
                tmp = target.with_name(target.name + '.partial')
                torch.save(episode, tmp)
                tmp.replace(target)
                entry.update(manifest_entry(entry['index'], episode, filename))
                entry.update(trajectory_sha256=sha256_file(target), trajectory_bytes=target.stat().st_size,
                             inference_metrics={k: v for k, v in episode['metrics'].items() if 'inference_latency' in k},
                             spl=episode['metrics']['habitat_lab']['spl'])
                if 'inference_latency_ms' in episode:
                    entry['inference_latency_ms'] = episode['inference_latency_ms'].tolist()
                movement_actions = int((episode['actions'] != int(Action.STOP)).sum())
                entry.update(movement_actions=movement_actions,
                             collision_rate_per_movement_action=int(episode['collisions'].sum()) / max(1, movement_actions))
                if videos:
                    entry['status'] = 'recorded'
                    write_json(path, manifest)
                    failure_stage = 'video_error'
                    video = export_video(episode, root / filename.replace('.pt', '.mp4'), nav.fps, composite=True)
                    entry.update(video=video.name, video_bytes=video.stat().st_size, video_sha256=sha256_file(video))
                entry['status'] = 'finished'
                attempt.update(status='finished', trajectory=filename, success=entry['success'], termination=entry['termination'])
            except Exception as exc:
                entry.update(status=failure_stage, error=f'{type(exc).__name__}: {exc}')
                attempt.update(status=entry['status'], error=entry['error'], trajectory=entry['trajectory'])
            entry['total_seconds'] = time.perf_counter() - started
            attempt['total_seconds'] = entry['total_seconds']
            write_json(path, manifest)
            print(f"{building(definition['source_scene_id'])}/{definition['source_episode_id']} status={entry['status']} success={entry['success']} error={entry['error']}", flush=True)
        # A video encode error does not undo a successfully recorded simulation.
        finished = [e for e in manifest['episodes'] if e['status'] in ('finished', 'video_error')]
        manifest['complete'] = all(e['status'] in ('finished', 'error', 'video_error') for e in manifest['episodes'])
        manifest['summary'] = dict(requested=len(definitions), executed=len(finished),
                                   successes=sum(e['success'] for e in finished),
                                   navigation_failures=sum(not e['success'] for e in finished),
                                   runtime_errors=sum(e['status'] == 'error' for e in manifest['episodes']),
                                   video_errors=sum(e['status'] == 'video_error' for e in manifest['episodes']),
                                   pending=sum(e['status'] in ('pending', 'running', 'recorded') for e in manifest['episodes']),
                                   success_rate_all_requested=sum(e['success'] for e in finished) / len(definitions),
                                   mean_spl_executed=float(np.mean([e['spl'] for e in finished])) if finished else None,
                                   collision_denominator='non-STOP actions, including turns',
                                   total_attempts=sum(len(e['attempts']) for e in manifest['episodes']))
        write_json(path, manifest)
        return manifest


def evaluate_policy(checkpoint, plan_path, phase, output_dir, encoder=None, resume=False, retry_errors=False,
                    max_steps=500, allow_integration=False):
    model, metadata = load_policy(checkpoint)
    plan = validate_splits(read_json(plan_path))
    if plan['fingerprint'] != metadata['splits']['fingerprint']:
        raise ValueError('Evaluation must use the saved training experiment split manifest')
    if metadata['integration_only'] and (not allow_integration or phase == 'final_evaluation'):
        raise ValueError('Tiny/test checkpoints may only run explicitly labeled integration rollouts, never final evaluation')
    if not metadata['integration_only'] and phase not in ('development', 'final_evaluation'):
        raise ValueError('Normal learned evaluation uses development or final_evaluation, not training buildings')
    sim = SimulatorConfig(**metadata['simulator_config'])
    nav = NavigationConfig(seed=metadata['train_config']['seed'], max_steps=max_steps, **metadata['navigation_config'])
    result = execute_plan(plan_path, phase, output_dir, sim, nav, learned_runner(model, metadata, encoder),
                          controller_identity=sha256_file(checkpoint), resume=resume, retry_errors=retry_errors,
                          videos=True, purpose='integration_rollout' if metadata['integration_only'] else 'learned_evaluation')
    return result

"""Small explicit commands for audit, features, behavior cloning and Habitat runs."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import json
import sys
import yaml
from .common import read_json
from .features import FeatureConfig
from .worker import WorkerClient, worker_command
from .training import TrainConfig


def feature_arguments(parser):
    parser.add_argument('--encoder-python', required=True, help='Python executable in the separate V-JEPA environment')
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--feature-config', type=Path, default=Path('configs/vjepa_features.yaml'))
    parser.add_argument('--encoder-device', default='cuda')
    parser.add_argument('--test-encoder', action='store_true', help='Local fixture only, never real V-JEPA features')


def make_worker(args):
    config = FeatureConfig(**yaml.safe_load(args.feature_config.read_text()))
    return WorkerClient(worker_command(args.encoder_python, config, args.source_dir, args.checkpoint,
                                       args.encoder_device, args.test_encoder))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('audit')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclusions', type=Path, help='JSON mapping Building/episode_id -> review reason')
    p.add_argument('--negligible-m', type=float, default=0.01)
    p = commands.add_parser('combine-audits')
    p.add_argument('--inputs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('inventory')
    p.add_argument('--episode-data', type=Path, nargs='+', required=True)
    p.add_argument('--scene-data-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('splits')
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--train-scenes', nargs='+', required=True)
    p.add_argument('--development-scenes', nargs='+', required=True)
    p.add_argument('--final-scenes', nargs='+', required=True)
    p.add_argument('--train-count', type=int, default=200)
    p.add_argument('--development-count', type=int, default=50)
    p.add_argument('--final-count', type=int, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('integration-split')
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('extract')
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--chunk-size', type=int, default=16)
    feature_arguments(p)
    p = commands.add_parser('train')
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/direct_policy.yaml'))
    p.add_argument('--tiny-steps', type=int)
    p.add_argument('--epochs', type=int)
    p.add_argument('--goal-only', action='store_true')
    p.add_argument('--allow-test-encoder', action='store_true')
    p.add_argument('--device', default='cpu')
    p = commands.add_parser('collect')
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--phase', choices=('train', 'development', 'final_evaluation'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--simulator-config', type=Path, default=Path('configs/simulator.yaml'))
    p.add_argument('--navigation-config', type=Path, default=Path('configs/gibson_pilot.yaml'))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--videos', action='store_true', help='Optional for larger expert collection')
    p = commands.add_parser('rollout')
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--splits', type=Path, required=True)
    p.add_argument('--phase', choices=('train', 'development', 'final_evaluation'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--allow-integration', action='store_true')
    p.add_argument('--max-steps', type=int, default=500)
    feature_arguments(p)
    p = commands.add_parser('check-encoder')
    feature_arguments(p)
    args = parser.parse_args(argv)
    try:
        if args.command == 'audit':
            from .audit import audit_run
            result = audit_run(args.run_dir, args.output, read_json(args.exclusions) if args.exclusions else None, args.negligible_m)
            print(json.dumps(result['summary'], indent=2))
        elif args.command == 'inventory':
            from .splits import inspect_availability
            result = inspect_availability(args.episode_data, args.scene_data_dir, args.output)
            print(json.dumps(result, indent=2))
            return 2 if result['errors'] else 0
        elif args.command == 'combine-audits':
            from .audit import combine_audits
            result = combine_audits(args.inputs, args.output)
            print(json.dumps(result['summary'], indent=2))
        elif args.command == 'splits':
            from .splits import prepare_splits
            result = prepare_splits(args.inventory,
                                    dict(train=args.train_scenes, development=args.development_scenes, final_evaluation=args.final_scenes),
                                    dict(train=args.train_count, development=args.development_count, final_evaluation=args.final_count),
                                    args.output, args.seed)
            print(f"Saved explicit split manifest: {args.output}; fingerprint={result['fingerprint']}")
        elif args.command == 'integration-split':
            from .splits import integration_split
            integration_split(args.audit, args.output)
            print(f'Pilot integration split: {args.output}; not a generalization experiment')
        elif args.command in ('extract', 'check-encoder'):
            with make_worker(args) as encoder:
                if args.command == 'extract':
                    from .cache import extract_cache
                    result = extract_cache(args.audit, args.cache_dir, encoder, args.chunk_size)
                    print(f"Cached {len(result['episodes'])} episodes; encoder={encoder.identity['backbone']}")
                else:
                    import numpy as np
                    config = FeatureConfig(**encoder.identity['feature_config'])
                    result = encoder.encode(np.zeros((config.history_length, 3, 224, 224), dtype=np.uint8))
                    print(json.dumps(dict(identity=encoder.identity, synthetic_probe_shape=list(result.shape),
                                          note='Encoder/loading/preprocessing check only, not real trajectory/navigation validation'), indent=2))
        elif args.command == 'train':
            from .training import train_policy
            values = yaml.safe_load(args.config.read_text())
            for key in ('tiny_steps', 'epochs'):
                if getattr(args, key) is not None:
                    values[key] = getattr(args, key)
            if args.goal_only:
                values['goal_only'] = True
            train_policy(args.cache_dir, args.splits, args.output_dir, TrainConfig(**values), args.device, args.allow_test_encoder)
        elif args.command == 'collect':
            from .execution import execute_plan
            from jepa_navigation.utils.config import load_config, SimulatorConfig, NavigationConfig
            result = execute_plan(args.splits, args.phase, args.output_dir,
                                  load_config(args.simulator_config, SimulatorConfig),
                                  load_config(args.navigation_config, NavigationConfig),
                                  resume=args.resume, retry_errors=args.retry_errors, videos=args.videos)
            print(json.dumps(result['summary'], indent=2))
            return 2 if result['summary']['runtime_errors'] or result['summary']['video_errors'] else (1 if result['summary']['navigation_failures'] else 0)
        else:
            from .execution import evaluate_policy
            from .training import load_policy
            model, _ = load_policy(args.policy)
            with nullcontext(None) if model.goal_only else make_worker(args) as encoder:
                result = evaluate_policy(args.policy, args.splits, args.phase, args.output_dir, encoder,
                                         args.resume, args.retry_errors, args.max_steps, args.allow_integration)
            print(json.dumps(result['summary'], indent=2))
            return 2 if result['summary']['runtime_errors'] or result['summary']['video_errors'] else (1 if result['summary']['navigation_failures'] else 0)
        return 0
    except (OSError, ValueError, RuntimeError, EOFError, KeyError, AssertionError) as exc:
        print(f'Baseline {args.command} error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

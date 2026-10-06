#!/usr/bin/env python3
"""Select available buildings, then optionally invoke the existing split writer.

No downloads, collection, training or evaluation. Final scenes are reserved;
external-baseline training/tuning overlap still requires a provenance review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PHASES = ('train', 'development', 'final_evaluation')
PILOT = {'adrian', 'albertville', 'anaheim'}


def select_buildings(inventory, config):
    if inventory.get('workflow') != 'official_asset_intersections' or inventory.get('errors'):
        raise ValueError('Resolve the inventory errors first.')
    seed = config['seed']
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Invalid seed')
    rows = inventory['buildings']
    if len({r['building'].casefold() for r in rows}) != len(rows):
        raise ValueError('Duplicate or case-ambiguous building names')
    used = PILOT | {n.casefold() for n in config['excluded_buildings']}
    final_excluded = {n.casefold() for n in config['final_excluded_buildings']}
    selected = {}
    for phase in PHASES:
        spec = config['splits'][phase]
        n, count = spec['buildings'], spec['episodes']
        expected_split = 'val' if phase == 'final_evaluation' else 'train'
        if spec['official_split'] != expected_split:
            raise ValueError('This protocol uses official train for train/development and val for final.')
        if type(n) is not int or type(count) is not int or n < 1 or count < n:
            raise ValueError('Each split needs positive buildings and at least one episode per building.')
        required = (count + n - 1) // n
        eligible = []
        for row in rows:
            name = row['building']
            if name.casefold() in used or not row['mesh_exists'] or len(row['inputs']) != 1:
                continue
            if phase == 'final_evaluation' and name.casefold() in final_excluded:
                continue
            source = row['inputs'][0]
            path = Path(source['episode_data'])
            if path.parent.name != expected_split or path.name != expected_split + '.json.gz':
                continue
            if source['count'] >= required:
                eligible.append(name)
        eligible.sort(key=lambda name: (hashlib.sha256(f'{seed}:{name.casefold()}'.encode()).hexdigest(), name))
        if len(eligible) < n:
            raise ValueError(f'{phase}: need {n} available {expected_split} buildings with at least '
                             f'{required} episodes each; found {len(eligible)}. Supply assets and rerun inventory; '
                             'do not silently reuse another split or reduce the budget.')
        selected[phase] = eligible[:n]
        used.update(name.casefold() for name in selected[phase])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/gibson_experiment.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--materialize', action='store_true', help='Also write exact episode definitions using the existing split CLI')
    args = parser.parse_args()
    raw_inventory = args.inventory.read_bytes()
    config = json.loads(args.config.read_text())
    groups = select_buildings(json.loads(raw_inventory), config)
    command = [sys.executable, str(Path(__file__).with_name('direct_baseline.py').resolve()),
               'splits', '--inventory', str(args.inventory.resolve()), '--seed', str(config['seed']),
               '--output', str((args.output_dir / 'experiment.json').resolve())]
    for phase, flag in zip(PHASES, ('train', 'development', 'final')):
        command += [f'--{flag}-scenes', *groups[phase], f'--{flag}-count', str(config['splits'][phase]['episodes'])]
    # Do not overwrite a previous selection, even if inventory availability changes.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selection = dict(config=config, buildings=groups,
                     inventory_sha256=hashlib.sha256(raw_inventory).hexdigest(),
                     final_status='reserved; external baseline provenance and compatibility pending',
                     materialize_command=command)
    (args.output_dir / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(groups, indent=2))
    if args.materialize:
        subprocess.run(command, check=True)
    else:
        print('To materialize the selected episodes:\n' + shlex.join(command))


if __name__ == '__main__':
    main()

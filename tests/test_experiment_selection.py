"""Building-selection safeguards only; fixtures are not research results."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('selection', ROOT / 'scripts/plan_gibson_experiment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/gibson_experiment.json').read_text())
        self.inventory = dict(workflow='official_asset_intersections', errors=[], buildings=[])
        for split, names in [('train', ['Adrian'] + [f'T{i}' for i in range(15)]),
                             ('val', ['Anaheim'] + [f'V{i}' for i in range(7)])]:
            for name in names:
                self.inventory['buildings'].append(dict(building=name, mesh_exists=True,
                    inputs=[dict(episode_data=f'/data/{split}/{split}.json.gz', count=100)]))

    def test_deterministic_disjoint_and_official_sources(self):
        result = module.select_buildings(self.inventory, self.config)
        self.inventory['buildings'].reverse()
        self.assertEqual(result, module.select_buildings(self.inventory, self.config))
        self.assertEqual([len(result[p]) for p in module.PHASES], [10, 3, 5])
        flat = sum(result.values(), [])
        self.assertEqual(len(flat), len(set(flat)))
        self.assertFalse(set(flat) & {'Adrian', 'Anaheim'})
        self.assertTrue(all(n.startswith('V') for n in result['final_evaluation']))

    def test_missing_assets_or_insufficient_capacity_do_not_fallback(self):
        for field, value in [('mesh_exists', False), ('count', 1)]:
            inventory = copy.deepcopy(self.inventory)
            for row in inventory['buildings']:
                if row['building'].startswith('V'):
                    if field == 'count':
                        row['inputs'][0][field] = value
                    else:
                        row[field] = value
            with self.assertRaisesRegex(ValueError, 'final_evaluation'):
                module.select_buildings(inventory, self.config)

    def test_baseline_exclusions_and_dirty_inventory(self):
        self.config['final_excluded_buildings'] = ['V0', 'V1']
        result = module.select_buildings(self.inventory, self.config)
        self.assertFalse(set(result['final_evaluation']) & {'V0', 'V1'})
        self.inventory['errors'] = ['unreadable shard']
        with self.assertRaises(ValueError):
            module.select_buildings(self.inventory, self.config)


if __name__ == '__main__':
    unittest.main()

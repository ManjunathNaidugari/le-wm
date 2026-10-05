#!/usr/bin/env python3
"""Verify real upstream config modules without importing Habitat's native runtime.

Run with Python 3.9/3.10, NumPy 1.26.4, Hydra 1.3.2 and OmegaConf 2.3.0.
This only checks schema composition; it cannot establish simulator compatibility.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from types import ModuleType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lab-source', type=Path, required=True, help='Habitat-Lab v0.3.3 checkout root')
    args = parser.parse_args()
    source = args.lab_source.resolve() / 'habitat-lab/habitat'
    if not (source / 'config/default_structured_configs.py').is_file():
        parser.error('Expected a Habitat-Lab source checkout')
    # Package namespaces load the upstream Python config files directly.
    # No simulator functions, observations or episode execution are substituted.
    for name, path in [('habitat', source), ('jepa_navigation', Path(__file__).resolve().parents[1] / 'src')]:
        package = ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    import habitat.config
    habitat = sys.modules['habitat']
    habitat.get_config = habitat.config.get_config
    from habitat.version import VERSION
    assert VERSION == '0.3.3', VERSION
    from jepa_navigation.simulator.lab_pointnav_env import build_lab_config
    from jepa_navigation.utils.config import SimulatorConfig, NavigationConfig
    from dataclasses import replace
    config = build_lab_config(habitat, SimulatorConfig(), NavigationConfig(), 0.2)
    assert list(config.habitat.simulator.agents.main_agent.sim_sensors) == ['rgb_sensor']
    assert list(config.habitat.task.actions) == ['stop', 'move_forward', 'turn_left', 'turn_right']
    assert config.habitat.task.measurements.success.success_distance == 0.2
    assert config.habitat.environment.max_episode_steps == 500
    assert config.habitat.simulator.turn_angle == 15
    for sim, nav in [(SimulatorConfig(), replace(NavigationConfig(), turn_degrees=15.5)),
                     (SimulatorConfig(hfov_degrees=90.5), NavigationConfig())]:
        try:
            build_lab_config(habitat, sim, nav, 0.2)
        except ValueError:
            pass
        else:
            raise AssertionError('Fractional integer-schema angles must be rejected')
    print(json.dumps(dict(result='pass', check='actual upstream configuration composition only',
                          python=sys.version, habitat_lab_source_version=VERSION,
                          packages={name: importlib.metadata.version(name) for name in
                                    ('numpy', 'hydra-core', 'omegaconf')},
                          native_runtime_verified=False), indent=2))


if __name__ == '__main__':
    main()

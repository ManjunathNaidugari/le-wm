#!/usr/bin/env python3
"""Run inside the NEW Linux environment before rendering any Gibson scene."""
import importlib.metadata
import json
import platform
import sys

from jepa_navigation.simulator.lab_pointnav_env import require_lab, build_lab_config
from jepa_navigation.navigation.actions import Action, from_lab_action
from jepa_navigation.utils.config import SimulatorConfig, NavigationConfig


def main():
    if sys.version_info[:2] != (3, 9) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Pinned runtime verification requires Linux x86_64 / Python 3.9')
    hs, habitat = require_lab()
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.utils.visualizations import maps
    config = build_lab_config(habitat, SimulatorConfig(), NavigationConfig(), 0.2)
    for name, expected in [('numpy', '1.26.4'), ('habitat-lab', '0.3.3'),
                           ('hydra-core', '1.3.2'), ('omegaconf', '2.3.0'), ('Pillow', '10.4.0')]:
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f'{name} must be {expected}')
    assert from_lab_action(HabitatSimActions.stop) == Action.STOP
    assert from_lab_action(HabitatSimActions.move_forward) == Action.FORWARD
    assert callable(maps.colorize_draw_agent_and_fit_to_height)
    assert config.habitat.task.measurements.success.success_distance == 0.2
    print(json.dumps(dict(habitat_sim=hs.__version__, habitat_lab=habitat.__version__,
                          python=sys.version, platform=platform.platform(),
                          configuration='composed successfully'), indent=2))


if __name__ == '__main__':
    main()

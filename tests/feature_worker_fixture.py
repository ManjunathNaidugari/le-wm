"""Subprocess protocol fixture, excluded from the installed application."""
from unittest.mock import patch
from baseline_fixtures import TestEncoder
from jepa_navigation.baseline.worker import main

if __name__ == '__main__':
    with patch('jepa_navigation.baseline.worker.official_encoder',
               side_effect=lambda source, checkpoint, config, device: TestEncoder(config)):
        raise SystemExit(main())

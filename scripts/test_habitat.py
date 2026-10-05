#!/usr/bin/env python3
"""Thin entry point; install the project with pip install -e . first."""
from jepa_navigation.utils.navigation_cli import verify_main


if __name__ == "__main__":
    raise SystemExit(verify_main())

"""
Demo: Configuration Manager
---------------------------

Demonstrates loading all project configuration files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import config_loader


def print_tree(data, indent=""):
    """
    Recursively print dictionaries and lists as a tree.
    """

    if isinstance(data, dict):

        items = list(data.items())

        for i, (key, value) in enumerate(items):

            is_last = i == len(items) - 1

            branch = "└── " if is_last else "├── "

            if isinstance(value, (dict, list)):
                print(f"{indent}{branch}{key}")
                print_tree(
                    value,
                    indent + ("    " if is_last else "│   ")
                )
            else:
                print(f"{indent}{branch}{key}: {value}")

    elif isinstance(data, list):

        for i, item in enumerate(data):

            is_last = i == len(data) - 1

            branch = "└── " if is_last else "├── "

            if isinstance(item, (dict, list)):
                print(f"{indent}{branch}")
                print_tree(
                    item,
                    indent + ("    " if is_last else "│   ")
                )
            else:
                print(f"{indent}{branch}{item}")


def main():

    print("=" * 60)
    print("           CONFIGURATION MANAGER DEMO")
    print("=" * 60)

    config_files = [
        "paths.yaml",
        "dataset.yaml",
        "model.yaml"
    ]

    for filename in config_files:

        print(f"\nLoading: {filename}")
        print("-" * 60)

        config = config_loader.load(filename)

        print_tree(config)

        print("\n✓ Loaded Successfully")

    print("\n" + "=" * 60)
    print("All configuration files loaded successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
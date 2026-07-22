"""
Demo: Dataset Validator
"""

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.validator import DatasetValidator


def main():

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    paths = config_loader.load("paths.yaml")
    dataset_root = paths["data"]["raw"] + "/WHU-Hi"

    discovery = DatasetDiscovery(dataset_root)
    scenes = discovery.discover()

    for scene in scenes:

        print(f"\nValidating : {scene['scene']}")

        try:
            DatasetValidator.validate(scene)
            print("Status     : PASSED")

        except Exception as e:
            print("Status     : FAILED")
            print(e)


if __name__ == "__main__":
    main()
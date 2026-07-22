"""
Demo: Dataset Loader
"""

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader


def main():

    print("=" * 60)
    print("DATASET LOADER")
    print("=" * 60)

    paths = config_loader.load("paths.yaml")

    discovery = DatasetDiscovery(
        paths["data"]["raw"] + "/WHU-Hi"
    )

    scenes = discovery.discover()

    for scene in scenes:

        dataset = DatasetLoader.load(scene)

        print(f"\nScene : {dataset['scene']}")
        print(f"Height : {dataset['height']}")
        print(f"Width : {dataset['width']}")
        print(f"Bands : {dataset['bands']}")
        print(f"Classes : {dataset['classes']}")
        print(f"Data Type : {dataset['dtype']}")


if __name__ == "__main__":
    main()
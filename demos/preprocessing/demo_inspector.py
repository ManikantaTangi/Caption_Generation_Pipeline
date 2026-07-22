"""
Demo: Dataset Inspector
"""

from pprint import pprint

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.inspector import DatasetInspector


def main():

    print("=" * 60)
    print("DATASET INSPECTOR")
    print("=" * 60)

    paths = config_loader.load("paths.yaml")

    discovery = DatasetDiscovery(
        paths["data"]["raw"] + "/WHU-Hi"
    )

    scenes = discovery.discover()

    for scene in scenes:

        dataset = DatasetLoader.load(scene)

        report = DatasetInspector.inspect(dataset)

        print("\n")
        pprint(report)


if __name__ == "__main__":
    main()
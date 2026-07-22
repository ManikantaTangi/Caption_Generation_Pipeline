"""
Demo: Dataset Discovery
-----------------------
Discovers all WHU-Hi dataset scenes.
"""

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery


def main():

    print("=" * 60)
    print("WHU-Hi Dataset Discovery")
    print("=" * 60)

    # Load configuration
    paths = config_loader.load("paths.yaml")

    # Dataset root directory
    dataset_root = paths["data"]["raw"] + "/WHU-Hi"

    # Discover scenes
    discovery = DatasetDiscovery(dataset_root)
    scenes = discovery.discover()

    print(f"\nDiscovered {len(scenes)} scene(s)\n")

    for i, scene in enumerate(scenes, start=1):

     print(f"\nScene {i}")
     print("-" * 60)
     print(f"Scene Name   : {scene['scene']}")

     print(f"HSI Cube     : {scene['cube']}")
     print(f"Exists       : {scene['cube'].exists()}")

     print(f"Ground Truth : {scene['ground_truth']}")
     print(f"Exists       : {scene['ground_truth'].exists()}")

     print(f"Split Folder : {scene['split_folder']}")
     print(f"Exists       : {scene['split_folder'].exists()}")

if __name__ == "__main__":
    main()
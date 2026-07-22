"""
Demo : Patch Generator
"""

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.patch_generator import PatchGenerator


def main():

    paths = config_loader.load("paths.yaml")

    discovery = DatasetDiscovery(
        paths["data"]["raw"] + "/WHU-Hi"
    )

    scene = discovery.discover()[0]

    dataset = DatasetLoader.load(scene)

    patches = PatchGenerator.generate(
        dataset,
        patch_size=32,
        stride=32
    )

    print("=" * 60)
    print("PATCH GENERATOR")
    print("=" * 60)

    print(f"Scene : {dataset['scene']}")
    print(f"Generated Patches : {len(patches)}")

    first = patches[0]

    print("\nFirst Patch")

    print("Cube Shape :", first["cube"].shape)

    print("GT Shape :", first["ground_truth"].shape)

    print("Location :", (first["row"], first["col"]))


if __name__ == "__main__":
    main()
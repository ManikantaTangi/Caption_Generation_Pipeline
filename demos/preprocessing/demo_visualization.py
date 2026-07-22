"""
Demo: Dataset Visualization
"""

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.visualization import DatasetVisualizer


def main():

    print("=" * 60)
    print("DATASET VISUALIZATION DEMO")
    print("=" * 60)

    paths = config_loader.load("paths.yaml")

    dataset_root = paths["data"]["raw"] + "/WHU-Hi"

    discovery = DatasetDiscovery(dataset_root)

    scenes = discovery.discover()

    for scene in scenes:

        print(f"\nVisualizing : {scene['scene']}")

        dataset = DatasetLoader.load(scene)

        DatasetVisualizer.show_rgb(dataset)

        DatasetVisualizer.show_ground_truth(dataset)

        DatasetVisualizer.show_single_band(
            dataset,
            band=10
        )

        DatasetVisualizer.show_band_histogram(
            dataset,
            band=10
        )

        DatasetVisualizer.compare_bands(
            dataset,
            bands=(10, 30, 50)
        )

        DatasetVisualizer.show_class_distribution(
            dataset
        )


if __name__ == "__main__":
    main()
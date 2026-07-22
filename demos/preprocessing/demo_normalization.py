"""
Demo : Dataset Normalization
"""

from src.utils.config import config_loader
from src.preprocessing.discovery import DatasetDiscovery
from src.preprocessing.loader import DatasetLoader
from src.preprocessing.normalization import DatasetNormalizer


def report(name, dataset):

    cube = dataset["cube"]

    print(f"\n{name}")

    print("-" * 40)

    print("Minimum :", cube.min())

    print("Maximum :", cube.max())

    print("Mean    :", cube.mean())

    print("Std     :", cube.std())


def main():

    paths = config_loader.load("paths.yaml")

    discovery = DatasetDiscovery(
        paths["data"]["raw"] + "/WHU-Hi"
    )

    scene = discovery.discover()[0]

    dataset = DatasetLoader.load(scene)

    report("Original", dataset)

    report(
        "Min-Max",
        DatasetNormalizer.min_max(dataset)
    )

    report(
        "Z-Score",
        DatasetNormalizer.z_score(dataset)
    )

    report(
        "Bandwise",
        DatasetNormalizer.bandwise(dataset)
    )

    report(
        "L2",
        DatasetNormalizer.l2(dataset)
    )


if __name__ == "__main__":
    main()
"""
Dataset Discovery Module
------------------------

Automatically discovers WHU-Hi scenes and their files.
"""

from pathlib import Path


class DatasetDiscovery:
    """
    Discover available HSI scenes inside the dataset directory.
    """

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def discover(self):
        """
        Scan dataset directory.

        Returns
        -------
        list
            List of discovered scene information.
        """

        scenes = []

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        for folder in sorted(self.root_dir.iterdir()):

            if not folder.is_dir():
                continue

            cube = None
            gt = None
            split_folder = None

            for file in folder.iterdir():

                if file.name.endswith("_gt.mat"):
                    gt = file

                elif file.name.endswith(".mat"):
                    cube = file

                elif file.is_dir():
                    split_folder = file

            scenes.append(
                {
                    "scene": folder.name,
                    "cube": cube,
                    "ground_truth": gt,
                    "split_folder": split_folder,
                }
            )

        return scenes

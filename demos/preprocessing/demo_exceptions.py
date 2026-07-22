"""
Demo : Custom Exceptions
"""

from src.preprocessing.exceptions import (
    DatasetNotFoundError,
    InvalidCubeShapeError,
    PatchGenerationError
)


def main():

    print("=" * 60)
    print("CUSTOM EXCEPTIONS DEMO")
    print("=" * 60)

    try:
        raise DatasetNotFoundError(
            "Dataset directory not found."
        )

    except DatasetNotFoundError as e:
        print(e)

    try:
        raise InvalidCubeShapeError(
            "Cube must be a 3D array."
        )

    except InvalidCubeShapeError as e:
        print(e)

    try:
        raise PatchGenerationError(
            "Unable to generate patches."
        )

    except PatchGenerationError as e:
        print(e)


if __name__ == "__main__":
    main()
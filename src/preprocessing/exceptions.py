"""
Custom Exceptions
-----------------

Custom exceptions used throughout the preprocessing pipeline.

Author: Manikanta Tangi
Project: HSI Caption Generation Pipeline
"""


class PreprocessingError(Exception):
    """
    Base class for all preprocessing exceptions.
    """

    pass


class DatasetNotFoundError(PreprocessingError):
    """
    Raised when the dataset directory cannot be found.
    """

    pass


class SceneNotFoundError(PreprocessingError):
    """
    Raised when a requested scene is unavailable.
    """

    pass


class CubeFileNotFoundError(PreprocessingError):
    """
    Raised when the hyperspectral cube file is missing.
    """

    pass


class GroundTruthFileNotFoundError(PreprocessingError):
    """
    Raised when the ground truth file is missing.
    """

    pass


class InvalidCubeShapeError(PreprocessingError):
    """
    Raised when the hyperspectral cube has an invalid shape.
    """

    pass


class InvalidGroundTruthError(PreprocessingError):
    """
    Raised when the ground truth has an invalid format.
    """

    pass


class DatasetValidationError(PreprocessingError):
    """
    Raised when dataset validation fails.
    """

    pass


class DatasetLoadingError(PreprocessingError):
    """
    Raised when dataset loading fails.
    """

    pass


class NormalizationError(PreprocessingError):
    """
    Raised when normalization fails.
    """

    pass


class PatchGenerationError(PreprocessingError):
    """
    Raised when patch generation fails.
    """

    pass


class VisualizationError(PreprocessingError):
    """
    Raised when visualization fails.
    """

    pass


class StatisticsError(PreprocessingError):
    """
    Raised when statistics computation fails.
    """

    pass
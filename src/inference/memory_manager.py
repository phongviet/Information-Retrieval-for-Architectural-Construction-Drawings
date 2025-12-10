import gc
import logging
from math import sqrt



class MemoryManager:
    """A utility class for managing memory safety during processing of large architectural drawings.

    This class provides methods to estimate image sizes in megapixels, check against safety thresholds
    for GPU memory constraints (typically 6GB), automatically downscale DPI using square root scaling
    to maintain quality, and perform explicit garbage collection to prevent memory accumulation.
    Designed for A0/A1 format drawings at high DPI on constrained systems.
    """

    def __init__(self) -> None:
        """Initialize the MemoryManager with logging setup."""
        self.logger = logging.getLogger(__name__)

    def calculate_megapixels(self, width: int, height: int) -> float:
        """Calculate the image size in megapixels.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Image size in megapixels as a float.
        """
        return (width * height) / 1_000_000

    def check_safety(self, width: int, height: int, max_mp: float) -> tuple[bool, int | None]:
        """Check if the image size is within the safe memory threshold.

        Calculates current megapixels and compares against the max_mp threshold.
        A0 drawings at 300 DPI ≈ 130 megapixels, so 200 MP threshold provides safety margin.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            max_mp: Maximum safe megapixels threshold.

        Returns:
            Tuple of (is_safe, recommended_dpi). is_safe is True if within threshold,
            recommended_dpi is None (calculated separately if needed).
        """
        current_mp = self.calculate_megapixels(width, height)
        if current_mp <= max_mp:
            return True, None
        else:
            return False, None

    def auto_downscale_dpi(self, current_dpi: int, current_mp: float, target_mp: float) -> int:
        """Calculate adjusted DPI using square root scaling for proportional reduction.

        Square root scaling preserves image quality better than linear scaling by maintaining aspect ratio.

        Args:
            current_dpi: Current DPI value.
            current_mp: Current megapixels.
            target_mp: Target megapixels threshold.

        Returns:
            New DPI value as integer.
        """
        new_dpi = int(current_dpi * sqrt(target_mp / current_mp))
        new_mp = current_mp * (new_dpi / current_dpi) ** 2
        self.logger.warning(
            f"Image {current_mp:.1f} MP exceeds threshold {target_mp} MP. "
            f"Auto-downscaling from {current_dpi} DPI to {new_dpi} DPI (estimated {new_mp:.1f} MP)"
        )
        return new_dpi

    def explicit_cleanup(self) -> None:
        """Perform explicit garbage collection to free unused memory.

        This method should be called after each page processing to prevent memory accumulation,
        especially important for GPU memory management on 6GB GTX 1660Ti.
        """
        gc.collect()
        self.logger.debug("Explicit memory cleanup executed")

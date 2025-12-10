import fitz  # PyMuPDF
import numpy as np
import logging


class PDFProcessor:
    """A class for processing PDF documents and converting pages to images.

    This class provides functionality to open PDF files, validate their integrity,
    and prepare them for high-quality rasterization at specified DPI levels for
    computer vision processing of architectural drawings.

    Attributes:
        doc (fitz.Document): The opened PDF document object.

    Example:
        processor = PDFProcessor("path/to/document.pdf")
        page_count = processor.get_page_count()
    """

    def __init__(self, pdf_path: str) -> None:
        """Initialize the PDFProcessor with a PDF file path.

        Opens the PDF document using PyMuPDF and stores the document reference
        for subsequent page operations. Validates that the file exists and is
        a valid PDF.

        Args:
            pdf_path: Path to the PDF file to process.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            fitz.FileDataError: If the PDF file is corrupt or cannot be opened.
        """
        self.logger = logging.getLogger(__name__)
        try:
            self.doc: fitz.Document = fitz.open(pdf_path)
        except FileNotFoundError as e:
            error_msg = f"PDF file not found at path: {pdf_path}. Please check the file path and ensure the file exists."
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg) from e
        except fitz.FileDataError as e:
            error_msg = f"Corrupt or invalid PDF file: {pdf_path}. The file may be damaged or not a valid PDF."
            self.logger.error(error_msg)
            raise fitz.FileDataError(error_msg) from e

    def get_page_count(self) -> int:
        """Return the total number of pages in the PDF document.

        Returns:
            The number of pages in the document.
        """
        return len(self.doc)

    def rasterize_page(self, page_num: int, dpi: int) -> np.ndarray:
        """Rasterize a specific page of the PDF at the given DPI.

        Extracts the specified page using zero-based indexing, calculates the
        scale factor to convert from PyMuPDF's default 72 DPI base to the target
        DPI, and creates a high-resolution pixmap using matrix transformation.
        The pixmap is then converted to a NumPy array in (height, width, 3) RGB
        format compatible with OpenCV and YOLO.

        Args:
            page_num: Zero-based index of the page to rasterize.
            dpi: Target DPI for rasterization.

        Returns:
            NumPy array of the rasterized page in (height, width, 3) RGB format.
        """
        if page_num < 0 or page_num >= len(self.doc):
            error_msg = f"Invalid page number {page_num}. Page number must be between 0 and {len(self.doc) - 1}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        page = self.doc[page_num]
        # Calculate scale factor to convert from PyMuPDF's 72 DPI base to target DPI
        scale = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        # Extract RGB pixel data from pixmap using pix.samples byte array
        # Reshape to (height, width, pix.n) dimensional array using pix.height, pix.width, and pix.n (number of channels)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        # Ensure RGB format (drop alpha channel if present, pix.n == 4 for RGBA)
        if pix.n == 4:
            img_array = img_array[:, :, :3]
        # The array is now in (height, width, 3) RGB format with uint8 data type, compatible with OpenCV and YOLO
        return img_array

import os
import cv2
import logging
from tqdm import tqdm

from .pdf_processor import PDFProcessor
from .memory_manager import MemoryManager
from .sahi_detector import SAHIDetector
from .utils import aggregate_detections
from .debug_utils import save_stitched_debug, visualize_overlap_regions


class InferencePipeline:
    """Orchestrates the complete inference pipeline for architectural drawing detection.

    Integrates PDF processing, memory management, SAHI detection, NMS deduplication,
    annotated image generation, and optional debug visualization.
    """

    def __init__(self, config: dict, output_dir: str = 'data/output') -> None:
        """Initialize the inference pipeline with configuration and output directory.

        Args:
            config: Configuration dictionary with all necessary parameters.
            output_dir: Directory to save output images and debug files.
        """
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Extract parameters
        self.dpi = config.get('dpi', 300)
        self.max_megapixels = config.get('max_megapixels', 200)
        self.model_path = config.get('model_path')
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.slice_height = config.get('slice_height', 640)
        self.slice_width = config.get('slice_width', 640)
        self.overlap_height_ratio = config.get('overlap_height_ratio', 0.2)
        self.overlap_width_ratio = config.get('overlap_width_ratio', 0.2)
        self.debug_mode = config.get('debug_mode', False)

        # Validate critical parameters
        if not self.model_path:
            raise ValueError("model_path is required in config for inference mode")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model path does not exist: {self.model_path}\n"
                f"For training mode, use: python main.py train --data dataset/data.yaml\n"
                f"For inference mode, ensure trained model exists at the specified path."
            )

        # Color mapping for annotations
        self.color_mapping = {
            'door': (0, 255, 0),    # Green
            'window': (0, 0, 255),  # Red (BGR)
            'wall': (255, 0, 0),    # Blue (BGR)
            'object': (0, 255, 255) # Yellow
        }

        # Prepare SAHI config
        self.sahi_config = {
            'confidence': self.confidence_threshold,
            'slice_height': self.slice_height,
            'slice_width': self.slice_width,
            'overlap_height_ratio': self.overlap_height_ratio,
            'overlap_width_ratio': self.overlap_width_ratio
        }

        # Slice params for debug
        self.slice_params = {
            'slice_height': self.slice_height,
            'slice_width': self.slice_width,
            'overlap_height_ratio': self.overlap_height_ratio,
            'overlap_width_ratio': self.overlap_width_ratio
        }

        # Initialize components
        self.memory_manager = MemoryManager()
        self.processor = None
        self.detector = None

        self.logger.info(
            f"InferencePipeline initialized: output_dir={output_dir}, DPI={self.dpi}, "
            f"SAHI slices={self.slice_height}x{self.slice_width} with {self.overlap_height_ratio} overlap"
        )

    def process_pdf(self, pdf_path: str) -> dict:
        """Process a PDF document through the complete inference pipeline.

        Args:
            pdf_path: Path to the PDF file to process.

        Returns:
            Dictionary with results per page.
        """
        self.processor = PDFProcessor(pdf_path)
        self.detector = SAHIDetector(self.model_path, self.sahi_config)

        page_count = self.processor.get_page_count()
        results = {}
        successful_pages = 0

        self.logger.info(f"Processing PDF: {pdf_path}, {page_count} pages")

        for page_num in tqdm(range(page_count), desc="Processing PDF", unit="page"):
            try:
                # Rasterize page
                page_image = self.processor.rasterize_page(page_num, self.dpi)
                h, w = page_image.shape[:2]

                # Memory safety check
                mp = self.memory_manager.calculate_megapixels(w, h)
                is_safe, _ = self.memory_manager.check_safety(w, h, self.max_megapixels)
                if not is_safe:
                    new_dpi = self.memory_manager.auto_downscale_dpi(self.dpi, mp, self.max_megapixels)
                    self.logger.warning(f"Page {page_num+1}: Downscaling from {self.dpi} to {new_dpi} DPI")
                    page_image = self.processor.rasterize_page(page_num, new_dpi)
                    h, w = page_image.shape[:2]  # Update dimensions

                # SAHI detection
                page_detections = self.detector.detect(page_image)

                # NMS deduplication
                page_detections = aggregate_detections([page_detections])

                # Generate annotated image
                img_copy = page_image.copy()
                for det in page_detections:
                    color = self.color_mapping.get(det['class_name'], (0, 255, 0))
                    bbox = det['bbox']
                    cv2.rectangle(img_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)

                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_copy, (int(bbox[0]), int(bbox[1]) - text_height - 5),
                                  (int(bbox[0]) + text_width, int(bbox[1])), (255, 255, 255), -1)
                    cv2.putText(img_copy, label, (int(bbox[0]), int(bbox[1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Save annotated image
                output_path = os.path.join(self.output_dir, f'page_{page_num+1}_detections.jpg')
                cv2.imwrite(output_path, img_copy, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Store results
                results[page_num+1] = {'detections': page_detections, 'annotated_image': output_path}
                successful_pages += 1

                # Debug mode
                if self.debug_mode:
                    save_stitched_debug(page_image, page_detections, page_num+1, self.output_dir)
                    visualize_overlap_regions(page_image, self.slice_params, self.output_dir, page_num+1)

                # Cleanup
                self.memory_manager.explicit_cleanup()

                self.logger.debug(f"Page {page_num+1} processed: {len(page_detections)} detections, annotated image saved")

            except Exception as e:
                self.logger.exception(f"Error processing page {page_num+1}: {e}")
                results[page_num+1] = {'error': str(e), 'annotated_image': None}

        self.logger.info(f"PDF processing complete: {successful_pages}/{page_count} pages processed successfully, annotated images saved to {self.output_dir}")

        # Close PDF
        if self.processor and self.processor.doc:
            self.processor.doc.close()

        return results

    def get_detector_info(self) -> dict:
        """Get detector configuration and architecture information.

        Returns:
            Dictionary containing detector configuration and architecture info.
            Returns empty dict if detector not initialized.
        """
        if self.detector:
            return self.detector.get_detector_info()
        return {}

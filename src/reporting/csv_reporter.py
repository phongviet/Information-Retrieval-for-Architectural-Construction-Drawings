import os
import logging
from typing import List, Dict
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('processing.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class CSVReporter:
    """
    A class for generating CSV reports from detection statistics data.

    This class handles directory validation, DataFrame conversion, CSV writing
    with proper formatting, and output validation to ensure data integrity.

    Attributes:
        output_dir (str): The directory where CSV reports will be saved.
    """

    def __init__(self, output_dir: str):
        """
        Initializes the CSVReporter with an output directory.

        Args:
            output_dir: Path to the directory for saving CSV reports.
                        Will be created if it does not exist.

        Raises:
            OSError: If the directory cannot be created.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Created output directory: {output_dir}")
        self.output_dir = output_dir
        logger.info(f"Initialized CSVReporter with output directory: {output_dir}")

    def generate_report(self, stats_data: List[Dict], output_filename: str = 'detection_report.csv') -> str:
        """
        Generates a CSV report from the provided statistics data.

        Args:
            stats_data: List of dictionaries containing statistics rows.
            output_filename: Name of the output CSV file. Defaults to 'detection_report.csv'.

        Returns:
            The full path to the generated CSV file.

        Raises:
            KeyError: If stats_data is missing expected columns.
            PermissionError: If unable to write to the output directory.
            IOError: For other file I/O issues.
        """
        logger.info(f"Generating CSV report: {output_filename}")
        try:
            logger.debug(f"Creating DataFrame from {len(stats_data)} rows")
            df = pd.DataFrame(stats_data)
            df = df[['Page_ID', 'Class', 'Count', 'Confidence_Avg', 'Min_Confidence', 'Max_Confidence']]
            output_path = os.path.join(self.output_dir, output_filename)
            logger.debug(f"Writing CSV to {output_path}")
            df.to_csv(output_path, index=False, float_format='%.3f')
            return output_path
        except KeyError as e:
            logger.exception(f"CSV structure invalid: missing expected columns - {e}")
            raise KeyError(f"Missing expected columns in stats_data: {e}") from e
        except (PermissionError, IOError) as e:
            logger.exception(f"Failed to write CSV file: {e}")
            raise IOError(f"Unable to write CSV to {self.output_dir}: {e}") from e

    def validate_output(self, csv_path: str) -> bool:
        """
        Validates the generated CSV file for existence and content.

        Args:
            csv_path: Path to the CSV file to validate.

        Returns:
            True if validation passes.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file is empty or has invalid structure.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if os.path.getsize(csv_path) == 0:
            raise ValueError("CSV file empty after write")
        try:
            df = pd.read_csv(csv_path)
            row_count = len(df)
            page_count = df['Page_ID'].nunique() if 'Page_ID' in df.columns else 0
            logger.info(f"Report generated: {csv_path}, {row_count} rows across {page_count} pages")
            return True
        except Exception as e:
            logger.exception(f"Error validating CSV: {e}")
            raise ValueError(f"CSV validation failed: {e}") from e

    def get_output_path(self, filename: str) -> str:
        """
        Constructs the full output path for a given filename.

        Args:
            filename: The name of the file.

        Returns:
            The full path to the file in the output directory.
        """
        return os.path.join(self.output_dir, filename)

import os
# Fix OMP: Error #15 - Allow duplicate OpenMP libraries (PyTorch, OpenCV, NumPy)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import logging
from src.cli.parser import create_parser, validate_architecture_args
from src.utils.validators import validate_environment, setup_logging
from src.utils.config_loader import ConfigLoader
from src.training.trainer import Trainer
from src.inference.pipeline import InferencePipeline
from src.reporting.stats_aggregator import structure_for_csv, aggregate_counts
from src.reporting.csv_reporter import CSVReporter
from src.reporting.metadata_generator import generate_metadata


def main():
    """Main entry point for the Architectural Drawing Detection Pipeline."""
    # Configure logging
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Parse command-line arguments
        parser = create_parser()
        args = parser.parse_args(sys.argv[1:])

        # Validate architecture arguments
        validate_architecture_args(args)

        command = args.command

        # Pre-flight validation
        validate_environment()
        logger.debug("Environment validation passed")

        # Load configuration
        config = ConfigLoader(args.config)

        if command == 'train':
            # Training mode - implement three-tier override strategy
            architecture = args.architecture if args.architecture else getattr(config, 'model_architecture', 'yolov8')
            variant = args.variant if args.variant else getattr(config, 'model_variant', 'nano')

            # Log architecture source
            source = 'CLI override' if args.architecture else 'config.yaml'
            logger.info(f"Training with: {architecture}{variant} (source: {source})")

            trainer = Trainer(config, architecture, variant)
            resume_checkpoint = args.resume if hasattr(args, 'resume') else None
            trainer.train(args.data, epochs=args.epochs, batch_size=args.batch_size, resume=resume_checkpoint)
            logger.info("Training mode complete")

        elif command == 'detect':
            # Detection mode
            if args.model:
                config.model['model_path'] = args.model
            elif not config.model.get('model_path'):
                # Default to models/custom_model.pt if not specified
                config.model['model_path'] = 'models/custom_model.pt'
                logger.info("No model_path specified, using default: models/custom_model.pt")

            # Merge config sections for pipeline (flatten nested structure)
            pipeline_config = {
                **config.model,
                **config.inference,
                'debug_mode': args.debug if hasattr(args, 'debug') else False
            }
            pipeline = InferencePipeline(pipeline_config, output_dir=args.output)
            results = pipeline.process_pdf(args.input)

            # Generate statistics
            stats_data = structure_for_csv(results)
            reporter = CSVReporter(args.output)
            csv_path = reporter.generate_report(stats_data)

            # Compute detection summary
            page_results = aggregate_counts(results)
            class_counts = {}
            for counts in page_results.values():
                for class_name, count in counts.items():
                    class_counts[class_name] = class_counts.get(class_name, 0) + count
            error_pages = sum(1 for page_data in results.values() if 'error' in page_data)
            detection_summary = {
                'total_pages': len(results),
                'class_counts': class_counts,
                'error_pages': error_pages
            }

            # Generate metadata
            architecture_info = pipeline.get_detector_info()
            metadata_path = generate_metadata(args.input, config.config, detection_summary, args.output, architecture_info)

            logger.info(f"Detection complete: CSV report and annotated images saved to {args.output}")

    except FileNotFoundError as e:
        sys.stderr.write(f"File not found: {e}. Verify file exists and path is correct.\n")
        logging.exception("Error in main execution")
        sys.exit(1)
    except RuntimeError as e:
        sys.stderr.write(f"Execution error: {e}. Check processing.log for details.\n")
        logging.exception("Error in main execution")
        sys.exit(1)
    except ValueError as e:
        sys.stderr.write(f"Invalid parameter: {e}.\n")
        logging.exception("Error in main execution")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}. See processing.log for full traceback.\n")
        logging.exception("Error in main execution")
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()

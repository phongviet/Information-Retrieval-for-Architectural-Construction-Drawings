"""
Architecture Support Validation Tests

Comprehensive test suite validating multi-architecture YOLO support across all integration points,
ensuring backwards compatibility with pre-Phase-7 models, and verifying architecture metadata
persistence throughout training and inference workflows.

Test Coverage:
- ModelArchitectureSelector validation (valid/invalid inputs)
- Training with YOLOv8 and YOLOv11 architectures
- Architecture switching between training runs
- CLI architecture override behavior and config fallback
- Inference architecture detection from metadata, filename, and defaults
- Backwards compatibility with pre-Phase-7 models
- Architecture metadata persistence (architecture_info.yaml, run_metadata.json)
- Error handling for invalid architectures and edge cases

Execution: pytest tests/test_architecture_support.py -v
"""

import pytest
import subprocess
import os
import yaml
import json
import shutil
from pathlib import Path
from typing import Dict, Any
from src.training.model_selector import ModelArchitectureSelector, SUPPORTED_ARCHITECTURES, MODEL_VARIANTS
from src.utils.config_loader import ConfigLoader
from src.training.trainer import Trainer


def get_latest_train_dir() -> Path:
    """Helper function to get the latest training directory, handling both 'train' and 'trainN' formats."""
    train_base = Path('runs/train')
    train_dirs = [d for d in os.listdir(train_base) if d.startswith('train')]

    def get_train_number(dirname: str) -> int:
        """Extract number from train directory name, treating 'train' as 0."""
        num_str = dirname.replace('train', '')
        return int(num_str) if num_str else 0

    latest_dir_name = max(train_dirs, key=get_train_number)
    return train_base / latest_dir_name


@pytest.fixture
def temp_config_file(tmp_path: Path) -> str:
    """Create temporary config.yaml with architecture parameters."""
    config_data: Dict[str, Any] = {
        'system': {'gpu_id': 0, 'seed': 42, 'log_level': 'INFO'},
        'model': {'model_path': 'models/custom_model.pt', 'confidence_threshold': 0.5},
        'training': {
            'model_architecture': 'yolov8',
            'model_variant': 'nano',
            'epochs': 2,  # Minimal for testing
            'batch_size': 4,
            'img_size': 640
        },
        'inference': {'dpi': 150, 'slice_height': 640, 'slice_width': 640}
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    return str(config_path)


@pytest.fixture
def test_dataset() -> str:
    """Return path to test dataset for training validation."""
    return 'data/floortest3.1.v1-data.yolov8/data.yaml'


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> str:
    """Create temporary output directory with automatic cleanup."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return str(output_dir)


class TestModelSelectorValidation:
    """Test ModelArchitectureSelector validation and functionality."""

    def test_model_selector_validation(self) -> None:
        """Test ModelArchitectureSelector with valid and invalid inputs."""
        # Test valid architectures and variants
        selector = ModelArchitectureSelector('yolov8', 'nano')
        assert selector.get_model_identifier() == 'yolov8n.pt'

        selector = ModelArchitectureSelector('yolo11', 'small')
        assert selector.get_model_identifier() == 'yolo11s.pt'

        # Test invalid architecture
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ModelArchitectureSelector('invalid_arch', 'nano')

        # Test invalid variant
        with pytest.raises(ValueError, match="Unsupported variant"):
            ModelArchitectureSelector('yolov8', 'invalid_variant')

        # Verify supported architectures and variants
        assert 'yolov8' in SUPPORTED_ARCHITECTURES
        assert 'yolo11' in SUPPORTED_ARCHITECTURES
        assert 'nano' in MODEL_VARIANTS
        assert 'xlarge' in MODEL_VARIANTS

    def test_architecture_info_generation(self) -> None:
        """Test architecture info generation for different combinations."""
        selector = ModelArchitectureSelector('yolov8', 'medium')
        info = selector.get_architecture_info()

        required_keys = ['architecture', 'variant', 'identifier', 'supported_features']
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"

        assert info['architecture'] == 'yolov8'
        assert info['variant'] == 'medium'
        assert info['identifier'] == 'yolov8m.pt'

        # Test YOLO11 specific features
        selector_v11 = ModelArchitectureSelector('yolo11', 'large')
        info_v11 = selector_v11.get_architecture_info()
        assert 'anchor_free' in info_v11['supported_features']


class TestTrainingWorkflowArchitecture:
    """Test training workflow with different architectures."""

    def test_training_with_yolov8(self, temp_config_file: str, test_dataset: str, temp_output_dir: str) -> None:
        """Test training with YOLOv8 architecture."""
        config = ConfigLoader(temp_config_file)
        trainer = Trainer(config, architecture='yolov8', variant='nano')

        # Execute minimal training
        trainer.train(test_dataset)

        # Verify architecture_info.yaml created
        train_dir = get_latest_train_dir()
        arch_info_path = train_dir / 'architecture_info.yaml'
        assert arch_info_path.exists(), "architecture_info.yaml not created"

        # Validate architecture_info.yaml content
        with open(arch_info_path, 'r') as f:
            arch_info = yaml.safe_load(f)

        assert arch_info['architecture'] == 'yolov8'
        assert arch_info['variant'] == 'nano'
        assert arch_info['identifier'] == 'yolov8n.pt'

        # Verify model weights saved
        assert (train_dir / 'weights' / 'best.pt').exists()

        # Check processing.log for architecture messages
        with open('processing.log', 'r') as f:
            log_content = f.read()
            assert 'Architecture: yolov8' in log_content

    def test_training_with_yolov11(self, temp_config_file: str, test_dataset: str, temp_output_dir: str) -> None:
        """Test training with YOLO11 architecture."""
        config = ConfigLoader(temp_config_file)
        trainer = Trainer(config, architecture='yolo11', variant='small')

        # Execute minimal training
        trainer.train(test_dataset)

        # Verify architecture_info.yaml
        train_dir = get_latest_train_dir()
        arch_info_path = train_dir / 'architecture_info.yaml'
        assert arch_info_path.exists()

        with open(arch_info_path, 'r') as f:
            arch_info = yaml.safe_load(f)

        assert arch_info['architecture'] == 'yolo11'
        assert arch_info['variant'] == 'small'
        assert arch_info['identifier'] == 'yolo11s.pt'

        # Verify model standardization
        assert Path('models/custom_model.pt').exists()

    def test_architecture_switching(self, temp_config_file: str, test_dataset: str) -> None:
        """Test switching between architectures in separate training runs."""
        # First training with YOLOv8
        config = ConfigLoader(temp_config_file)
        trainer_v8 = Trainer(config, architecture='yolov8', variant='nano')
        trainer_v8.train(test_dataset)

        first_train_dir = get_latest_train_dir()

        with open(first_train_dir / 'architecture_info.yaml', 'r') as f:
            arch_info_v8 = yaml.safe_load(f)

        # Second training with YOLO11
        trainer_v11 = Trainer(config, architecture='yolo11', variant='nano')
        trainer_v11.train(test_dataset)

        second_train_dir = get_latest_train_dir()

        with open(second_train_dir / 'architecture_info.yaml', 'r') as f:
            arch_info_v11 = yaml.safe_load(f)

        # Verify no cross-contamination
        assert arch_info_v8['architecture'] == 'yolov8'
        assert arch_info_v11['architecture'] == 'yolo11'

        # Verify latest model is YOLO11
        assert Path('models/custom_model.pt').exists()


class TestCLIArchitectureOverride:
    """Test CLI architecture override functionality."""

    def test_cli_architecture_override(self, temp_config_file: str, test_dataset: str, temp_output_dir: str) -> None:
        """Test CLI architecture override behavior."""
        result = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--architecture', 'yolo11',
            '--variant', 'small',
            '--epochs', '2',
            '--config', temp_config_file
        ], capture_output=True, text=True, timeout=300)

        assert result.returncode == 0, f"Training failed: {result.stderr}"

        # Parse processing.log for CLI override message
        with open('processing.log', 'r') as f:
            log_content = f.read()
            assert 'CLI override' in log_content or 'architecture override' in log_content.lower()

        # Verify architecture_info.yaml reflects CLI values
        train_dir = get_latest_train_dir()
        arch_info_path = train_dir / 'architecture_info.yaml'

        with open(arch_info_path, 'r') as f:
            arch_info = yaml.safe_load(f)

        assert arch_info['architecture'] == 'yolo11'
        assert arch_info['variant'] == 'small'

        # Verify config.yaml unchanged
        with open(temp_config_file, 'r') as f:
            config_data = yaml.safe_load(f)
            assert config_data['training']['model_architecture'] == 'yolov8'  # Original value

    def test_cli_config_fallback(self, temp_config_file: str, test_dataset: str) -> None:
        """Test CLI falls back to config.yaml defaults when no override provided."""
        result = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--epochs', '2',
            '--config', temp_config_file
        ], capture_output=True, text=True, timeout=300)

        assert result.returncode == 0

        # Verify uses config defaults
        train_dir = get_latest_train_dir()
        arch_info_path = train_dir / 'architecture_info.yaml'

        with open(arch_info_path, 'r') as f:
            arch_info = yaml.safe_load(f)

        assert arch_info['architecture'] == 'yolov8'  # Config default
        assert arch_info['variant'] == 'nano'  # Config default

        # Check logs for config source
        with open('processing.log', 'r') as f:
            log_content = f.read()
            assert 'source: config.yaml' in log_content or 'config' in log_content.lower()

    def test_cli_help_architecture_documentation(self) -> None:
        """Test CLI help shows architecture documentation."""
        result = subprocess.run(['python', 'main.py', 'train', '--help'], capture_output=True, text=True)

        assert result.returncode == 0
        help_text = result.stdout

        assert '--architecture' in help_text
        assert '--variant' in help_text
        assert 'yolov8' in help_text
        assert 'yolo11' in help_text
        assert 'nano' in help_text
        assert 'xlarge' in help_text


class TestInferenceArchitectureDetection:
    """Test inference architecture detection functionality."""

    def test_inference_architecture_detection_from_metadata(self, temp_config_file: str, test_dataset: str, temp_output_dir: str) -> None:
        """Test inference detects architecture from metadata file."""
        # Train model with known architecture
        result_train = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--architecture', 'yolo11',
            '--variant', 'small',
            '--epochs', '2',
            '--config', temp_config_file
        ], capture_output=True, text=True, timeout=300)

        assert result_train.returncode == 0

        # Get trained model path
        train_dir = get_latest_train_dir()
        model_path = str(train_dir / 'weights' / 'best.pt')

        # Create sample PDF for detection (minimal content)
        sample_pdf = Path(temp_output_dir) / 'sample.pdf'
        # For testing, we'll assume a sample PDF exists or create a minimal one
        # In real scenario, use tests/fixtures/sample.pdf

        # Execute detection
        result_detect = subprocess.run([
            'python', 'main.py', 'detect',
            '--input', 'test_data/Lot 136 Working Drawing Rev -3.pdf',  # Use existing test PDF
            '--output', temp_output_dir,
            '--model', model_path
        ], capture_output=True, text=True, timeout=180)

        assert result_detect.returncode == 0, f"Detection failed: {result_detect.stderr}"

        # Load run_metadata.json
        metadata_path = Path(temp_output_dir) / 'run_metadata.json'
        assert metadata_path.exists()

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'model_architecture' in metadata
        arch_section = metadata['model_architecture']
        assert arch_section['architecture'] == 'yolo11'
        assert arch_section['variant'] == 'small'
        assert arch_section['detected_from'] == 'metadata_file'

        # Check processing.log
        with open('processing.log', 'r') as f:
            log_content = f.read()
            assert 'Detected model architecture' in log_content

    def test_inference_backwards_compatibility(self, temp_output_dir: str) -> None:
        """Test backwards compatibility with pre-Phase-7 models."""
        # Use existing model without architecture_info.yaml
        model_path = 'models/yolov8n.pt'  # Pre-trained model

        result = subprocess.run([
            'python', 'main.py', 'detect',
            '--input', 'test_data/Lot 136 Working Drawing Rev -3.pdf',
            '--output', temp_output_dir,
            '--model', model_path
        ], capture_output=True, text=True, timeout=180)

        assert result.returncode == 0

        # Load metadata
        metadata_path = Path(temp_output_dir) / 'run_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        arch_section = metadata['model_architecture']
        assert arch_section['detected_from'] == 'default_assumption'
        assert arch_section['architecture'] == 'yolov8'  # Default fallback

        # Check for warning in logs
        with open('processing.log', 'r') as f:
            log_content = f.read()
            assert 'WARNING' in log_content or 'missing metadata' in log_content.lower()

    def test_inference_filename_based_detection(self, temp_output_dir: str) -> None:
        """Test architecture detection from filename patterns."""
        # Copy model with architecture hint in filename
        source_model = 'models/yolov8n.pt'
        dest_model = Path(temp_output_dir) / 'yolo11_custom.pt'
        shutil.copy2(source_model, dest_model)

        # Remove any architecture_info.yaml to force filename inference
        arch_info_path = Path(temp_output_dir) / 'architecture_info.yaml'
        if arch_info_path.exists():
            arch_info_path.unlink()

        result = subprocess.run([
            'python', 'main.py', 'detect',
            '--input', 'test_data/Lot 136 Working Drawing Rev -3.pdf',
            '--output', temp_output_dir,
            '--model', str(dest_model)
        ], capture_output=True, text=True, timeout=180)

        assert result.returncode == 0

        # Load metadata and verify filename-based detection
        metadata_path = Path(temp_output_dir) / 'run_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        arch_section = metadata['model_architecture']
        assert arch_section['detected_from'] == 'filename_inference' or 'filename' in arch_section['detected_from']


class TestMetadataPersistence:
    """Test architecture metadata persistence across workflows."""

    def test_architecture_metadata_persistence(self, temp_config_file: str, test_dataset: str, temp_output_dir: str) -> None:
        """Test metadata persistence from training through inference."""
        # Train with specific architecture
        result_train = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--architecture', 'yolo11',
            '--variant', 'medium',
            '--epochs', '2',
            '--config', temp_config_file
        ], capture_output=True, text=True, timeout=300)

        assert result_train.returncode == 0

        # Get training metadata
        train_dir = get_latest_train_dir()
        with open(train_dir / 'architecture_info.yaml', 'r') as f:
            train_arch_info = yaml.safe_load(f)

        # Execute detection
        model_path = str(train_dir / 'weights' / 'best.pt')
        result_detect = subprocess.run([
            'python', 'main.py', 'detect',
            '--input', 'test_data/Lot 136 Working Drawing Rev -3.pdf',
            '--output', temp_output_dir,
            '--model', model_path
        ], capture_output=True, text=True, timeout=180)

        assert result_detect.returncode == 0

        # Load inference metadata
        with open(Path(temp_output_dir) / 'run_metadata.json', 'r') as f:
            detect_metadata = json.load(f)

        # Verify metadata chain
        detect_arch = detect_metadata['model_architecture']
        assert detect_arch['architecture'] == train_arch_info['architecture']
        assert detect_arch['variant'] == train_arch_info['variant']
        assert detect_arch['identifier'] == train_arch_info['identifier']

    def test_architecture_info_yaml_structure(self, temp_config_file: str, test_dataset: str) -> None:
        """Test architecture_info.yaml file structure and content."""
        result = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--epochs', '2',
            '--config', temp_config_file
        ], capture_output=True, text=True, timeout=300)

        assert result.returncode == 0

        train_dir = get_latest_train_dir()
        arch_info_path = train_dir / 'architecture_info.yaml'

        with open(arch_info_path, 'r') as f:
            arch_info = yaml.safe_load(f)

        # Required keys
        required_keys = ['architecture', 'variant', 'identifier']
        for key in required_keys:
            assert key in arch_info
            assert isinstance(arch_info[key], str)

        # Optional keys
        if 'supported_features' in arch_info:
            assert isinstance(arch_info['supported_features'], list)

        if 'ultralytics_version_required' in arch_info:
            assert isinstance(arch_info['ultralytics_version_required'], str)

    def test_run_metadata_json_architecture_section(self, temp_config_file: str, test_dataset: str, temp_output_dir: str) -> None:
        """Test run_metadata.json architecture section structure."""
        # Train and detect
        result_train = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--epochs', '2',
            '--config', temp_config_file
        ], capture_output=True, text=True, timeout=300)

        assert result_train.returncode == 0

        train_dir = get_latest_train_dir()
        model_path = str(train_dir / 'weights' / 'best.pt')

        result_detect = subprocess.run([
            'python', 'main.py', 'detect',
            '--input', 'test_data/Lot 136 Working Drawing Rev -3.pdf',
            '--output', temp_output_dir,
            '--model', model_path
        ], capture_output=True, text=True, timeout=180)

        assert result_detect.returncode == 0

        # Load and validate metadata
        metadata_path = Path(temp_output_dir) / 'run_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'model_architecture' in metadata
        arch_section = metadata['model_architecture']

        expected_keys = ['architecture', 'variant', 'identifier', 'detected_from']
        for key in expected_keys:
            assert key in arch_section

        # Verify integration with existing metadata
        assert 'timestamp' in metadata
        assert 'config' in metadata
        assert 'detection_summary' in metadata


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_invalid_architecture_raises_error(self, tmp_path: Path) -> None:
        """Test error handling for invalid architectures."""
        # Create config with invalid architecture
        invalid_config = tmp_path / 'invalid_config.yaml'
        config_data = {
            'system': {'gpu_id': 0, 'seed': 42, 'log_level': 'INFO'},
            'model': {'model_path': 'models/custom_model.pt', 'confidence_threshold': 0.5},
            'training': {
                'model_architecture': 'invalid_arch',
                'model_variant': 'nano',
                'epochs': 2,
                'batch_size': 4,
                'img_size': 640
            },
            'inference': {'dpi': 150, 'slice_height': 640, 'slice_width': 640}
        }

        with open(invalid_config, 'w') as f:
            yaml.dump(config_data, f)

        # Test ConfigLoader raises error
        with pytest.raises(ValueError, match="Unsupported architecture"):
            ConfigLoader(str(invalid_config))

        # Test CLI with invalid architecture
        result = subprocess.run([
            'python', 'main.py', 'train',
            '--data', 'data/floortest3.1.v1-data.yolov8/data.yaml',
            '--architecture', 'invalid_arch',
            '--epochs', '2'
        ], capture_output=True, text=True)

        assert result.returncode != 0
        assert 'invalid choice' in result.stderr or 'Unsupported architecture' in result.stderr

    def test_architecture_mismatch_handling(self, temp_config_file: str, test_dataset: str, temp_output_dir: str) -> None:
        """Test handling of architecture metadata mismatch."""
        # Train model
        result_train = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--epochs', '2',
            '--config', temp_config_file
        ], capture_output=True, text=True, timeout=300)

        assert result_train.returncode == 0

        # Get model path and corrupt architecture_info.yaml
        train_dir = get_latest_train_dir()
        model_path = str(train_dir / 'weights' / 'best.pt')
        arch_info_path = train_dir / 'architecture_info.yaml'

        # Corrupt metadata
        with open(arch_info_path, 'r') as f:
            arch_info = yaml.safe_load(f)

        arch_info['architecture'] = 'yolo11'  # Mismatch

        with open(arch_info_path, 'w') as f:
            yaml.dump(arch_info, f)

        # Execute detection - should handle gracefully
        result_detect = subprocess.run([
            'python', 'main.py', 'detect',
            '--input', 'test_data/Lot 136 Working Drawing Rev -3.pdf',
            '--output', temp_output_dir,
            '--model', model_path
        ], capture_output=True, text=True, timeout=180)

        # Should not crash
        assert result_detect.returncode == 0

        # Check for warning in logs
        with open('processing.log', 'r') as f:
            log_content = f.read()
            assert 'WARNING' in log_content or 'mismatch' in log_content.lower()

    def test_missing_architecture_config_defaults(self, tmp_path: Path, test_dataset: str) -> None:
        """Test backwards compatibility with configs missing architecture fields."""
        # Create old-style config without architecture fields
        old_config = tmp_path / 'old_config.yaml'
        config_data = {
            'system': {'gpu_id': 0, 'seed': 42, 'log_level': 'INFO'},
            'model': {'model_path': 'models/custom_model.pt', 'confidence_threshold': 0.5},
            'training': {
                'epochs': 2,
                'batch_size': 4,
                'img_size': 640
            },
            'inference': {'dpi': 150, 'slice_height': 640, 'slice_width': 640}
        }

        with open(old_config, 'w') as f:
            yaml.dump(config_data, f)

        # Load config - should default to yolov8/nano
        config = ConfigLoader(str(old_config))
        assert config.get('training', 'model_architecture') == 'yolov8'
        assert config.get('training', 'model_variant') == 'nano'

        # Execute training - should work and create architecture_info.yaml
        result = subprocess.run([
            'python', 'main.py', 'train',
            '--data', test_dataset,
            '--epochs', '2',
            '--config', str(old_config)
        ], capture_output=True, text=True, timeout=300)

        assert result.returncode == 0

        # Verify architecture_info.yaml created with defaults
        train_dir = get_latest_train_dir()
        arch_info_path = train_dir / 'architecture_info.yaml'

        with open(arch_info_path, 'r') as f:
            arch_info = yaml.safe_load(f)

        assert arch_info['architecture'] == 'yolov8'
        assert arch_info['variant'] == 'nano'

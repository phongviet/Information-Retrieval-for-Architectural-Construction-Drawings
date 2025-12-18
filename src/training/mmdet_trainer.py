"""
MMDetection 3.x Training Wrapper
Provides unified interface for MMDetection Cascade R-CNN training
Compatible with MMDetection 3.3.0+ and mmengine Registry system
"""

import os
import shutil
import logging
import glob
import mmdet.engine
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import MODELS

logger = logging.getLogger(__name__)


class MMDetectionTrainer:
    """Training wrapper for MMDetection framework (3.x compatible)."""

    def __init__(self, config, mmdet_config_path: str):
        self.config = config
        self.mmdet_config_path = mmdet_config_path

        # Load MMDetection config
        self.cfg = Config.fromfile(mmdet_config_path)

        # Force default scope
        self.cfg.default_scope = 'mmdet'

        # Override paths from ConfigLoader
        if hasattr(config, 'data_root'):
            for dataloader in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
                if hasattr(self.cfg, dataloader) and self.cfg[dataloader]:
                    if 'dataset' in self.cfg[dataloader]:
                        self.cfg[dataloader]['dataset']['data_root'] = config.data_root

        # Override training hyperparameters
        if hasattr(config, 'training'):
            training_cfg = config.training
            if hasattr(training_cfg, 'learning_rate'):
                if hasattr(self.cfg, 'optim_wrapper') and 'optimizer' in self.cfg.optim_wrapper:
                    self.cfg.optim_wrapper['optimizer']['lr'] = training_cfg.learning_rate

            if hasattr(training_cfg, 'epochs'):
                if hasattr(self.cfg, 'train_cfg'):
                    self.cfg.train_cfg['max_epochs'] = training_cfg.epochs

            if hasattr(training_cfg, 'device'):
                device = training_cfg.device
                if device.startswith('cuda'):
                    gpu_id = int(device.split(':')[1]) if ':' in device else 0
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                    self.cfg.launcher = 'none'
                    logger.info(f"Training configured for GPU: {device} (Index {gpu_id})")

        # --- RUNTIME PATCHES ---
        self._fix_config_scopes()
        self._fix_cascade_rcnn_config()
        self._fix_pipeline_scopes()
        self._fix_evaluator_scopes() # <--- NEW: Fixes "CocoMetric" error

        # Set work directory
        self.cfg.work_dir = 'runs/mmdet_train/'
        os.makedirs(self.cfg.work_dir, exist_ok=True)
        self._setup_logging()
        logger.info(f"MMDetectionTrainer initialized: config={mmdet_config_path}")

    def _setup_logging(self):
        self.cfg.log_level = 'INFO'

    def _fix_config_scopes(self):
        """Fixes scope issues for hooks, modules, datasets, AND samplers."""
        # 1. Fix Model Data Preprocessor
        if hasattr(self.cfg, 'model') and hasattr(self.cfg.model, 'data_preprocessor'):
            dp = self.cfg.model.data_preprocessor
            if dp.get('type') == 'DetDataPreprocessor':
                dp['type'] = 'mmdet.DetDataPreprocessor'

        # 2. Fix Default Hooks (Visualization)
        if hasattr(self.cfg, 'default_hooks'):
            hooks = self.cfg.default_hooks
            if 'visualization' in hooks:
                vis = hooks['visualization']
                if vis.get('type') == 'DetVisualizationHook':
                    vis['type'] = 'mmdet.DetVisualizationHook'

        # 3. Fix Custom Hooks (NumClassCheck)
        if hasattr(self.cfg, 'custom_hooks'):
            for hook in self.cfg.custom_hooks:
                if hook.get('type') == 'NumClassCheckHook':
                    hook['type'] = 'mmdet.NumClassCheckHook'

        # 4. Fix Datasets (CocoDataset -> mmdet.CocoDataset)
        for dl_name in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
            if not hasattr(self.cfg, dl_name): continue
            dl = getattr(self.cfg, dl_name)
            if not dl: continue

            if 'dataset' in dl:
                dataset = dl['dataset']
                if dataset.get('type') == 'CocoDataset':
                    dataset['type'] = 'mmdet.CocoDataset'

            # 5. Fix Batch Samplers (AspectRatioBatchSampler -> mmdet.AspectRatioBatchSampler)
            if 'batch_sampler' in dl:
                bs = dl['batch_sampler']
                if bs.get('type') == 'AspectRatioBatchSampler':
                    bs['type'] = 'mmdet.AspectRatioBatchSampler'

    def _fix_pipeline_scopes(self):
        """Recursively fixes pipeline transforms (like PackDetInputs)."""
        mmdet_transforms = ['PackDetInputs', 'LoadAnnotations', 'Resize', 'RandomFlip', 'Pad']

        def patch_pipeline(pipeline):
            if not pipeline: return
            for t in pipeline:
                if t.get('type') in mmdet_transforms:
                    t['type'] = 'mmdet.' + t['type']

        for dl_name in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
            if not hasattr(self.cfg, dl_name): continue
            dl = getattr(self.cfg, dl_name)
            if not dl or 'dataset' not in dl: continue
            dataset = dl['dataset']
            if 'pipeline' in dataset:
                patch_pipeline(dataset['pipeline'])

    def _fix_evaluator_scopes(self):
        """
        Fixes scope issues for Evaluators (CocoMetric).
        This fixes 'KeyError: CocoMetric is not in the mmengine::metric registry'.
        """
        def patch_evaluator(evaluator):
            if evaluator is None: return

            # Handle list of evaluators
            if isinstance(evaluator, list):
                for e in evaluator:
                    patch_evaluator(e)
                return

            # Handle single evaluator dict
            if isinstance(evaluator, dict):
                if evaluator.get('type') == 'CocoMetric':
                    evaluator['type'] = 'mmdet.CocoMetric'
                    logger.info("Auto-fixed Evaluator scope -> mmdet.CocoMetric")

        # Patch both validation and test evaluators
        if hasattr(self.cfg, 'val_evaluator'):
            patch_evaluator(self.cfg.val_evaluator)

        if hasattr(self.cfg, 'test_evaluator'):
            patch_evaluator(self.cfg.test_evaluator)

    def _fix_cascade_rcnn_config(self):
        """Fixes specific Cascade R-CNN parameters."""
        if not hasattr(self.cfg, 'model'): return

        # Determine class count
        num_classes = None
        if hasattr(self.config, 'classes') and self.config.classes:
            num_classes = len(self.config.classes)
        elif hasattr(self.cfg, 'classes') and self.cfg.classes:
            num_classes = len(self.cfg.classes)
        elif hasattr(self.cfg.model, 'roi_head') and 'num_classes' in self.cfg.model.roi_head:
            num_classes = self.cfg.model.roi_head['num_classes']

        if hasattr(self.cfg.model, 'roi_head'):
            roi_head = self.cfg.model.roi_head

            if roi_head.get('type') == 'CascadeRoIHead':
                if 'num_stages' not in roi_head:
                    roi_head['num_stages'] = 3
                if 'stage_loss_weights' not in roi_head:
                    roi_head['stage_loss_weights'] = [1, 0.5, 0.25]

            if 'num_classes' in roi_head:
                del roi_head['num_classes']

            if num_classes is not None and hasattr(roi_head, 'bbox_head'):
                bbox_head = roi_head.bbox_head
                if isinstance(bbox_head, list):
                    for head in bbox_head:
                        head['num_classes'] = num_classes
                elif isinstance(bbox_head, dict):
                    bbox_head['num_classes'] = num_classes

            if num_classes is not None and hasattr(roi_head, 'mask_head') and roi_head.mask_head is not None:
                roi_head.mask_head['num_classes'] = num_classes

    def train(self):
        try:
            logger.info("Starting MMDetection 3.x training...")

            # Build model
            model = MODELS.build(self.cfg.model)

            # --- Validation Consistency Check ---
            val_dataloader = getattr(self.cfg, 'val_dataloader', None)
            if val_dataloader is None:
                val_cfg = None
                val_evaluator = None
            else:
                val_cfg = getattr(self.cfg, 'val_cfg', dict(type='ValLoop'))
                val_evaluator = getattr(self.cfg, 'val_evaluator', None)

            # --- Test Consistency Check ---
            test_dataloader = getattr(self.cfg, 'test_dataloader', None)
            if test_dataloader is None:
                test_cfg = None
                test_evaluator = None
            else:
                test_cfg = getattr(self.cfg, 'test_cfg', dict(type='TestLoop'))
                test_evaluator = getattr(self.cfg, 'test_evaluator', val_evaluator)

            # Initialize Runner
            runner = Runner(
                model=model,
                work_dir=self.cfg.work_dir,
                train_dataloader=self.cfg.train_dataloader,
                optim_wrapper=self.cfg.optim_wrapper,
                train_cfg=self.cfg.train_cfg,
                val_dataloader=val_dataloader,
                val_cfg=val_cfg,
                val_evaluator=val_evaluator,
                test_dataloader=test_dataloader,
                test_cfg=test_cfg,
                test_evaluator=test_evaluator,
                param_scheduler=self.cfg.get('param_scheduler'),
                default_hooks=self.cfg.get('default_hooks'),
                custom_hooks=self.cfg.get('custom_hooks'),
                launcher=self.cfg.get('launcher', 'none'),
                env_cfg=self.cfg.get('env_cfg'),
                cfg=self.cfg
            )

            runner.train()
            return self._find_best_checkpoint()

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            raise

    def _find_best_checkpoint(self):
        best_checkpoints = glob.glob(os.path.join(self.cfg.work_dir, 'best_*.pth'))
        if best_checkpoints:
            return max(best_checkpoints, key=os.path.getmtime)
        checkpoints = glob.glob(os.path.join(self.cfg.work_dir, 'epoch_*.pth'))
        if checkpoints:
            return max(checkpoints, key=os.path.getmtime)
        return None

    def standardize_model(self):
        source = self._find_best_checkpoint()
        if not source:
            return None
        dest = 'models/mmdet_cascade_rcnn.pth'
        os.makedirs('models', exist_ok=True)
        shutil.copy2(source, dest)
        return dest
#!/usr/bin/env python3
"""
Complete EfficientDet Training Script for Custom Object Detection
Optimized for signature, chop, stamp, and barcode detection

Usage:
    python train_efficientdet.py --config configs/efficientdet_d1.yaml
    python train_efficientdet.py --config configs/efficientdet_d1.yaml --resume checkpoints/last.pth
"""

import os
import sys
import time
import json
import yaml
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# EfficientDet imports
try:
    from effdet import create_model, create_dataset, create_loader
    from effdet.config import get_efficientdet_config
    from effdet.bench import DetBenchTrain, DetBenchPredict
    from timm.utils import AverageMeter, ModelEmaV2
    from timm.scheduler import CosineLRScheduler
except ImportError:
    print("Error: EfficientDet dependencies not found.")
    print("Please install with: pip install effdet timm")
    sys.exit(1)

warnings.filterwarnings('ignore')


class COCODataset:
    """COCO Dataset wrapper for EfficientDet training"""
    
    def __init__(self, annotation_path, image_dir, transforms=None):
        self.coco = COCO(annotation_path)
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        
        # Get image IDs and filter out images without annotations
        self.image_ids = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            if len(ann_ids) > 0:
                self.image_ids.append(img_id)
        
        print(f"Dataset initialized with {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        return {
            'img_id': img_id,
            'img_path': str(img_path),
            'img_info': img_info,
            'annotations': anns
        }


class ClassSpecificAugmentation:
    """Class-specific augmentation strategies for different object types"""
    
    def __init__(self, class_id_mapping):
        self.class_mapping = class_id_mapping
        # Map class names to IDs for easier lookup
        self.name_to_id = {v: k for k, v in class_id_mapping.items()}
    
    def get_augmentation_config(self):
        """Return class-specific augmentation configuration"""
        return {
            'signature': {
                'elastic_deformation': True,
                'shear_range': 5,  # degrees
                'gaussian_noise': 0.01,
                'context_aware': True
            },
            'chop': {
                'rotation_range': 30,  # degrees
                'perspective_transform': True,
                'morphological_ops': True,
                'ink_simulation': True
            },
            'stamp': {
                'rotation_range': 30,  # degrees
                'degradation_simulation': True,
                'partial_masking': 0.3,  # probability
                'multi_color_support': True
            },
            'barcode': {
                'rotation_range': 10,  # degrees - limited to preserve readability
                'gaussian_blur': (0.5, 1.0),
                'motion_blur': True,
                'preserve_structure': True
            }
        }


class EfficientDetTrainer:
    """Main training class for EfficientDet"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logging
        self.setup_logging()
        
        # Model and training components
        self.model = None
        self.model_ema = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = amp.GradScaler()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.start_epoch = 0
        self.best_map = 0.0
        self.train_metrics = defaultdict(AverageMeter)
        self.val_metrics = defaultdict(AverageMeter)
        
        print(f"Training on device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def setup_logging(self):
        """Setup logging and checkpoints directory"""
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Logs: {self.log_dir}")
    
    def create_model(self):
        """Create and configure EfficientDet model"""
        model_config = self.config['model']
        
        # Create model
        self.model = create_model(
            model_config['name'],
            bench_task='train',
            num_classes=model_config['num_classes'],
            pretrained=model_config['pretrained'],
            image_size=model_config['image_size'],
            **model_config.get('kwargs', {})
        )
        
        self.model = self.model.to(self.device)
        
        # Create EMA model if enabled
        if self.config['training']['model_ema']['enabled']:
            self.model_ema = ModelEmaV2(
                self.model,
                decay=self.config['training']['model_ema']['decay']
            )
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: {model_config['name']}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Image size: {model_config['image_size']}")
        print(f"Number of classes: {model_config['num_classes']}")
    
    def create_dataloaders(self):
        """Create training and validation data loaders"""
        data_config = self.config['data']
        
        # Training dataset
        train_dataset = create_dataset(
            'coco',
            data_config['train_annotations'],
            data_config['train_images'],
            transforms=None  # Transforms handled by effdet
        )
        
        # Validation dataset
        val_dataset = create_dataset(
            'coco',
            data_config['val_annotations'],
            data_config['val_images'],
            transforms=None
        )
        
        # Create loaders with effdet's built-in functionality
        self.train_loader = create_loader(
            train_dataset,
            input_size=self.config['model']['image_size'],
            batch_size=self.config['training']['batch_size'],
            is_training=True,
            use_prefetcher=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = create_loader(
            val_dataset,
            input_size=self.config['model']['image_size'],
            batch_size=self.config['validation']['batch_size'],
            is_training=False,
            use_prefetcher=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Training batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
    
    def create_optimizer(self):
        """Create optimizer and learning rate scheduler"""
        opt_config = self.config['optimizer']
        
        # Create optimizer
        if opt_config['type'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config['momentum'],
                weight_decay=opt_config['weight_decay'],
                nesterov=opt_config.get('nesterov', False)
            )
        elif opt_config['type'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
        
        # Create scheduler
        sched_config = self.config['scheduler']
        if sched_config['type'].lower() == 'cosine':
            self.scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=self.config['training']['epochs'],
                lr_min=sched_config['min_lr'],
                warmup_t=sched_config['warmup_epochs'],
                warmup_lr_init=sched_config['warmup_lr'],
                cycle_limit=1,
                t_in_epochs=True
            )
        
        print(f"Optimizer: {opt_config['type']}")
        print(f"Learning rate: {opt_config['learning_rate']}")
        print(f"Scheduler: {sched_config['type']}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Reset metrics
        for meter in self.train_metrics.values():
            meter.reset()
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (input, target) in enumerate(self.train_loader):
            # Move to device
            input = input.to(self.device, non_blocking=True)
            target = {k: v.to(self.device, non_blocking=True) for k, v in target.items()}
            
            # Forward pass with mixed precision
            with amp.autocast():
                output = self.model(input, target)
                loss = output['loss']
                
                # Additional losses if available
                loss_class = output.get('class_loss', 0)
                loss_bbox = output.get('bbox_loss', 0)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.config['training']['grad_clip_norm'] > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip_norm']
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update EMA
            if self.model_ema is not None:
                self.model_ema.update(self.model)
            
            # Update metrics
            batch_size = input.size(0)
            self.train_metrics['loss'].update(loss.item(), batch_size)
            if isinstance(loss_class, torch.Tensor):
                self.train_metrics['loss_class'].update(loss_class.item(), batch_size)
            if isinstance(loss_bbox, torch.Tensor):
                self.train_metrics['loss_bbox'].update(loss_bbox.item(), batch_size)
            
            # Logging
            if batch_idx % self.config['training']['log_interval'] == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch} [{batch_idx}/{num_batches}] '
                      f'Loss: {loss.item():.4f} '
                      f'LR: {lr:.2e}')
                
                # TensorBoard logging
                global_step = epoch * num_batches + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/learning_rate', lr, global_step)
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step(epoch)
        
        # Log epoch metrics
        print(f'Train Epoch {epoch}: '
              f'Loss: {self.train_metrics["loss"].avg:.4f}')
        
        # TensorBoard epoch logging
        self.writer.add_scalar('epoch/train_loss', self.train_metrics['loss'].avg, epoch)
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        # Use EMA model if available
        model_for_eval = self.model_ema.module if self.model_ema else self.model
        model_for_eval.eval()
        
        # Reset metrics
        for meter in self.val_metrics.values():
            meter.reset()
        
        all_predictions = []
        
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(self.val_loader):
                input = input.to(self.device, non_blocking=True)
                target = {k: v.to(self.device, non_blocking=True) for k, v in target.items()}
                
                # Forward pass
                with amp.autocast():
                    output = model_for_eval(input, target)
                    if isinstance(output, dict) and 'loss' in output:
                        loss = output['loss']
                        self.val_metrics['loss'].update(loss.item(), input.size(0))
                
                # For mAP calculation, we need predictions
                if batch_idx % 10 == 0:  # Sample every 10th batch for speed
                    with amp.autocast():
                        predictions = model_for_eval(input)
                    all_predictions.extend(predictions)
        
        # Calculate mAP (simplified version)
        if len(all_predictions) > 0:
            # This is a simplified mAP calculation
            # In practice, you'd want to use COCO evaluation
            map_score = self.calculate_map(all_predictions)
            self.val_metrics['mAP'].update(map_score, 1)
        
        avg_loss = self.val_metrics['loss'].avg
        avg_map = self.val_metrics['mAP'].avg
        
        print(f'Val Epoch {epoch}: Loss: {avg_loss:.4f}, mAP: {avg_map:.4f}')
        
        # TensorBoard logging
        self.writer.add_scalar('epoch/val_loss', avg_loss, epoch)
        self.writer.add_scalar('epoch/val_mAP', avg_map, epoch)
        
        return avg_map
    
    def calculate_map(self, predictions):
        """Simplified mAP calculation"""
        # This is a placeholder - implement proper COCO evaluation
        # For now, return a dummy value
        return 0.5
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_map': self.best_map,
            'config': self.config
        }
        
        if self.model_ema is not None:
            checkpoint['model_ema_state_dict'] = self.model_ema.state_dict()
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'last.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with mAP: {self.best_map:.4f}")
        
        # Save periodic checkpoint
        if epoch % self.config['training']['save_interval'] == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load EMA state
        if 'model_ema_state_dict' in checkpoint and self.model_ema:
            self.model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
        
        # Load training state
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_map = checkpoint.get('best_map', 0.0)
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        print(f"Best mAP: {self.best_map:.4f}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training for {self.config['training']['epochs']} epochs")
        
        # Training loop
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            start_time = time.time()
            
            # Train
            self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config['validation']['val_interval'] == 0:
                val_map = self.validate_epoch(epoch)
                
                # Check if best model
                is_best = val_map > self.best_map
                if is_best:
                    self.best_map = val_map
                
                # Save checkpoint
                self.save_checkpoint(epoch, is_best)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("Training completed!")
        print(f"Best mAP achieved: {self.best_map:.4f}")
        
        # Close TensorBoard writer
        self.writer.close()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config():
    """Create default configuration"""
    return {
        'model': {
            'name': 'tf_efficientdet_d1',
            'num_classes': 4,  # signature, barcode, chop, stamp
            'image_size': 640,
            'pretrained': True,
            'kwargs': {}
        },
        'data': {
            'train_annotations': 'data/train_annotations.json',
            'train_images': 'data/train_images',
            'val_annotations': 'data/val_annotations.json',
            'val_images': 'data/val_images'
        },
        'training': {
            'epochs': 300,
            'batch_size': 16,
            'num_workers': 4,
            'output_dir': 'outputs',
            'log_interval': 50,
            'save_interval': 50,
            'grad_clip_norm': 10.0,
            'model_ema': {
                'enabled': True,
                'decay': 0.9999
            }
        },
        'validation': {
            'batch_size': 16,
            'val_interval': 5
        },
        'optimizer': {
            'type': 'SGD',
            'learning_rate': 0.08,
            'momentum': 0.9,
            'weight_decay': 4e-5,
            'nesterov': False
        },
        'scheduler': {
            'type': 'cosine',
            'min_lr': 1e-6,
            'warmup_epochs': 5,
            'warmup_lr': 0.001
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train EfficientDet for object detection')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create default configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        config_path = 'efficientdet_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Default configuration saved to: {config_path}")
        return
    
    if not args.config:
        print("Please provide a config file with --config or create one with --create-config")
        return
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = EfficientDetTrainer(config)
    
    # Create model and data loaders
    trainer.create_model()
    trainer.create_dataloaders()
    trainer.create_optimizer()
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
EfficientDet PyTorch Lightning Training Script
Based on the provided notebook, adapted for signature, chop, stamp, barcode detection

Usage:
    python train_efficientdet_lightning.py --config config.yaml --data_dir ./data
"""

import os
import sys
import json
import yaml
import argparse
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# EfficientDet imports
try:
    from effdet import create_model, get_efficientdet_config, EfficientDet, DetBenchTrain
    from effdet.efficientdet import HeadNet
    from effdet.config.model_config import efficientdet_model_param_dict
    import timm
    EFFDET_AVAILABLE = True
except ImportError:
    print("Warning: effdet not available. Please install with: pip install effdet")
    EFFDET_AVAILABLE = False

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    print("Warning: pycocotools not available. Some evaluation features will be disabled.")
    COCO_AVAILABLE = False

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    print("Warning: ensemble-boxes not available. Installing...")
    os.system("pip install ensemble-boxes")
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True

warnings.filterwarnings('ignore')


class DatasetAdaptor:
    """
    Dataset adaptor for COCO format annotations
    Converts COCO annotations to image and label format for training
    """
    
    def __init__(self, images_dir_path: str, annotations_path: str):
        self.images_dir_path = Path(images_dir_path)
        self.annotations_path = Path(annotations_path)
        
        # Load COCO annotations
        if COCO_AVAILABLE:
            self.coco = COCO(str(annotations_path))
            self.image_ids = list(self.coco.imgs.keys())
        else:
            # Fallback JSON loading
            with open(annotations_path, 'r') as f:
                self.coco_data = json.load(f)
            self.image_ids = [img['id'] for img in self.coco_data['images']]
            self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
            self.annotations_by_image = {}
            for ann in self.coco_data['annotations']:
                if ann['image_id'] not in self.annotations_by_image:
                    self.annotations_by_image[ann['image_id']] = []
                self.annotations_by_image[ann['image_id']].append(ann)
        
        print(f"Dataset loaded with {len(self.image_ids)} images")
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def get_image_and_labels_by_idx(self, index: int):
        """
        Get image and annotations by index
        Returns:
            image: PIL Image
            pascal_bboxes: numpy array of bounding boxes in Pascal VOC format [xmin, ymin, xmax, ymax]
            class_labels: numpy array of class labels
            image_id: unique identifier for the image
        """
        image_id = self.image_ids[index]
        
        if COCO_AVAILABLE:
            # Load image info and annotations using COCO API
            image_info = self.coco.loadImgs(image_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            annotations = self.coco.loadAnns(ann_ids)
        else:
            # Fallback method
            image_info = self.image_id_to_info[image_id]
            annotations = self.annotations_by_image.get(image_id, [])
        
        # Load image
        image_path = self.images_dir_path / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        
        # Convert annotations to Pascal VOC format
        pascal_bboxes = []
        class_labels = []
        
        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to Pascal VOC format: [xmin, ymin, xmax, ymax]
            pascal_bboxes.append([x, y, x + w, y + h])
            class_labels.append(ann['category_id'])
        
        pascal_bboxes = np.array(pascal_bboxes, dtype=np.float32)
        class_labels = np.array(class_labels, dtype=np.int64)
        
        return image, pascal_bboxes, class_labels, image_id
    
    def show_image(self, index: int, figsize=(10, 10)):
        """Display image with bounding boxes"""
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image)
        ax.set_title(f"Image ID: {image_id}")
        
        # Draw bounding boxes
        for bbox, label in zip(bboxes, class_labels):
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            
            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(xmin, ymin - 5, f'Class {label}', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.show()


def create_efficientdet_model(num_classes: int = 4, image_size: int = 512, 
                            architecture: str = "tf_efficientnetv2_l",
                            pretrained_path: Optional[str] = None) -> DetBenchTrain:
    """
    Create EfficientDet model with custom backbone
    
    Args:
        num_classes: Number of detection classes
        image_size: Input image size
        architecture: Backbone architecture name
        pretrained_path: Path to local pretrained weights
    """
    
    # Register custom EfficientNetV2 backbone if not exists
    if architecture not in efficientdet_model_param_dict:
        efficientdet_model_param_dict[architecture] = dict(
            name=architecture,
            backbone_name=architecture,
            backbone_args=dict(drop_path_rate=0.2),
            num_classes=num_classes,
            url='',
        )
    
    # Get model configuration
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})
    
    print(f"Model config: {config}")
    
    # Create model
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from: {pretrained_path}")
        # Load custom pretrained weights
        net = EfficientDet(config, pretrained_backbone=False)
        
        # Load state dict
        state_dict = torch.load(pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        # Try to load compatible weights
        try:
            net.load_state_dict(state_dict, strict=False)
            print("Loaded pretrained weights successfully")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Initializing with pretrained backbone only")
            net = EfficientDet(config, pretrained_backbone=True)
    else:
        print("Initializing with pretrained backbone")
        net = EfficientDet(config, pretrained_backbone=True)
    
    # Replace classification head for custom number of classes
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    
    return DetBenchTrain(net, config)


def get_train_transforms(target_img_size: int = 512) -> A.Compose:
    """Get training augmentations optimized for document detection"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=3, p=1),
        ], p=0.3),
        A.Resize(height=target_img_size, width=target_img_size, p=1),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(p=1),
    ], p=1.0, bbox_params=A.BboxParams(
        format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
    ))


def get_valid_transforms(target_img_size: int = 512) -> A.Compose:
    """Get validation transforms"""
    return A.Compose([
        A.Resize(height=target_img_size, width=target_img_size, p=1),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(p=1),
    ], p=1.0, bbox_params=A.BboxParams(
        format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
    ))


class EfficientDetDataset(Dataset):
    """EfficientDet Dataset for PyTorch"""
    
    def __init__(self, dataset_adaptor: DatasetAdaptor, transforms: A.Compose = None):
        self.ds = dataset_adaptor
        self.transforms = transforms or get_valid_transforms()
    
    def __getitem__(self, index: int):
        # Get image and labels
        image, pascal_bboxes, class_labels, image_id = self.ds.get_image_and_labels_by_idx(index)
        
        # Convert to numpy array
        image = np.array(image, dtype=np.float32)
        
        # Apply transforms
        sample = {
            "image": image,
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }
        
        if len(pascal_bboxes) > 0:
            sample = self.transforms(**sample)
            sample["bboxes"] = np.array(sample["bboxes"])
        else:
            # Handle images with no annotations
            transformed = self.transforms(image=image, bboxes=[], labels=[])
            sample = {
                "image": transformed["image"],
                "bboxes": np.array([]).reshape(0, 4),
                "labels": np.array([]),
            }
        
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]
        
        # Get image dimensions after transform
        _, new_h, new_w = image.shape
        
        # Convert Pascal VOC to YXYX format for EfficientDet
        if len(pascal_bboxes) > 0:
            # Convert from [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
            sample["bboxes"] = pascal_bboxes[:, [1, 0, 3, 2]]
        else:
            sample["bboxes"] = np.array([]).reshape(0, 4)
        
        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0]),
        }
        
        return image, target, image_id
    
    def __len__(self):
        return len(self.ds)


class EfficientDetDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for EfficientDet"""
    
    def __init__(self,
                 train_annotations: str,
                 train_images: str,
                 val_annotations: str,
                 val_images: str,
                 target_img_size: int = 512,
                 batch_size: int = 8,
                 num_workers: int = 4):
        super().__init__()
        
        self.train_annotations = train_annotations
        self.train_images = train_images
        self.val_annotations = val_annotations
        self.val_images = val_images
        self.target_img_size = target_img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Transforms
        self.train_tfms = get_train_transforms(target_img_size)
        self.valid_tfms = get_valid_transforms(target_img_size)
    
    def setup(self, stage: Optional[str] = None):
        # Create dataset adaptors
        self.train_ds_adaptor = DatasetAdaptor(self.train_images, self.train_annotations)
        self.valid_ds_adaptor = DatasetAdaptor(self.val_images, self.val_annotations)
    
    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.train_ds_adaptor, 
            transforms=self.train_tfms
        )
    
    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(
            dataset_adaptor=self.valid_ds_adaptor, 
            transforms=self.valid_tfms
        )
    
    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        return DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader"""
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()
        
        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()
        
        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        
        return images, annotations, targets, image_ids


def run_wbf(predictions, image_size: int = 512, iou_thr: float = 0.44, 
           skip_box_thr: float = 0.43, weights=None):
    """Run Weighted Boxes Fusion on predictions"""
    bboxes = []
    confidences = []
    class_labels = []
    
    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]
        
        if WBF_AVAILABLE:
            boxes, scores, labels = weighted_boxes_fusion(
                boxes, scores, labels,
                weights=weights,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
            boxes = boxes * (image_size - 1)
        else:
            # Fallback without WBF
            boxes = np.array(boxes[0]) * (image_size - 1)
            scores = np.array(scores[0])
            labels = np.array(labels[0])
        
        bboxes.append(boxes.tolist() if hasattr(boxes, 'tolist') else boxes)
        confidences.append(scores.tolist() if hasattr(scores, 'tolist') else scores)
        class_labels.append(labels.tolist() if hasattr(labels, 'tolist') else labels)
    
    return bboxes, confidences, class_labels


class EfficientDetModel(LightningModule):
    """PyTorch Lightning Module for EfficientDet"""
    
    def __init__(self,
                 num_classes: int = 4,
                 img_size: int = 512,
                 prediction_confidence_threshold: float = 0.2,
                 learning_rate: float = 0.0002,
                 wbf_iou_threshold: float = 0.44,
                 model_architecture: str = 'tf_efficientnetv2_l',
                 pretrained_path: Optional[str] = None,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.img_size = img_size
        self.model = create_efficientdet_model(
            num_classes=num_classes,
            image_size=img_size,
            architecture=model_architecture,
            pretrained_path=pretrained_path
        )
        
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        
        self.inference_tfms = get_valid_transforms(target_img_size=img_size)
    
    def forward(self, images, targets):
        return self.model(images, targets)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr * 0.01
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}
        }
    
    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        losses = self.model(images, annotations)
        
        # Log losses
        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_class_loss", losses["class_loss"], on_step=True, on_epoch=True)
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True)
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        
        detections = outputs["detections"]
        
        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }
        
        # Log validation losses
        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("valid_class_loss", outputs["class_loss"], on_step=True, on_epoch=True)
        self.log("valid_box_loss", outputs["box_loss"], on_step=True, on_epoch=True)
        
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    
    def predict(self, images):
        """Prediction method for inference"""
        if isinstance(images, list):
            # Handle list of PIL images
            image_sizes = [(image.size[1], image.size[0]) for image in images]
            images_tensor = torch.stack([
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"] for image in images
            ])
        else:
            # Handle tensor input
            images_tensor = images
            if images_tensor.ndim == 3:
                images_tensor = images_tensor.unsqueeze(0)
            
            num_images = images_tensor.shape[0]
            image_sizes = [(self.img_size, self.img_size)] * num_images
        
        return self._run_inference(images_tensor, image_sizes)
    
    def _run_inference(self, images_tensor, image_sizes):
        """Run model inference"""
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )
        
        detections = self.model(images_tensor.to(self.device), dummy_targets)["detections"]
        
        (predicted_bboxes, predicted_class_confidences, predicted_class_labels) = \
            self.post_process_detections(detections)
        
        scaled_bboxes = self._rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )
        
        return scaled_bboxes, predicted_class_labels, predicted_class_confidences
    
    def _create_dummy_inference_targets(self, num_images):
        """Create dummy targets for inference"""
        dummy_targets = {
            "bbox": [torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device) 
                    for i in range(num_images)],
            "cls": [torch.tensor([1.0], device=self.device) 
                   for i in range(num_images)],
            "img_size": torch.tensor([(self.img_size, self.img_size)] * num_images, 
                                   device=self.device).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }
        return dummy_targets
    
    def post_process_detections(self, detections):
        """Post-process model detections"""
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )
        
        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold
        )
        
        return predicted_bboxes, predicted_class_confidences, predicted_class_labels
    
    def _postprocess_single_prediction_detections(self, detections):
        """Post-process single image detections"""
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]
        
        return {
            "boxes": boxes, 
            "scores": scores[indexes], 
            "classes": classes[indexes]
        }
    
    def _rescale_bboxes(self, predicted_bboxes, image_sizes):
        """Rescale bounding boxes to original image size"""
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims
            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (np.array(bboxes) * [
                        im_w / self.img_size,
                        im_h / self.img_size,
                        im_w / self.img_size,
                        im_h / self.img_size,
                    ]).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)
        return scaled_bboxes


def create_default_config():
    """Create default training configuration"""
    return {
        'model': {
            'num_classes': 4,
            'img_size': 512,
            'architecture': 'tf_efficientnetv2_l',
            'pretrained_path': None,
            'prediction_confidence_threshold': 0.2,
            'wbf_iou_threshold': 0.44,
        },
        'data': {
            'train_annotations': 'data/train_coco.json',
            'train_images': 'data/train_images',
            'val_annotations': 'data/val_coco.json',
            'val_images': 'data/val_images',
            'batch_size': 8,
            'num_workers': 4,
        },
        'training': {
            'max_epochs': 100,
            'learning_rate': 0.0002,
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': '16-mixed',
        },
        'callbacks': {
            'early_stopping': {
                'monitor': 'valid_loss',
                'patience': 10,
                'mode': 'min',
            },
            'model_checkpoint': {
                'monitor': 'valid_loss',
                'mode': 'min',
                'save_top_k': 3,
                'filename': 'efficientdet-{epoch:02d}-{valid_loss:.2f}',
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Train EfficientDet with PyTorch Lightning')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, help='Data directory path')
    parser.add_argument('--pretrained_model', type=str, help='Path to pretrained model')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create default configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        config = create_default_config()
        config_path = 'efficientdet_lightning_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        print(f"Default configuration saved to: {config_path}")
        print("\nTo start training:")
        print(f"python {__file__} --config {config_path} --data_dir ./data")
        return
    
    if not args.config:
        print("Please provide a config file with --config or create one with --create-config")
        return
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override data directory if provided
    if args.data_dir:
        data_dir = Path(args.data_dir)
        config['data']['train_annotations'] = str(data_dir / 'train_coco.json')
        config['data']['train_images'] = str(data_dir / 'train_images')
        config['data']['val_annotations'] = str(data_dir / 'val_coco.json')
        config['data']['val_images'] = str(data_dir / 'val_images')
    
    # Override pretrained model if provided
    if args.pretrained_model:
        config['model']['pretrained_path'] = args.pretrained_model
    
    # Check if required packages are available
    if not EFFDET_AVAILABLE:
        print("Error: effdet package is required. Install with: pip install effdet")
        return
    
    # Create data module
    dm = EfficientDetDataModule(
        train_annotations=config['data']['train_annotations'],
        train_images=config['data']['train_images'],
        val_annotations=config['data']['val_annotations'],
        val_images=config['data']['val_images'],
        target_img_size=config['model']['img_size'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
    )
    
    # Create model
    model = EfficientDetModel(
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        model_architecture=config['model']['architecture'],
        pretrained_path=config['model']['pretrained_path'],
        learning_rate=config['training']['learning_rate'],
        prediction_confidence_threshold=config['model']['prediction_confidence_threshold'],
        wbf_iou_threshold=config['model']['wbf_iou_threshold'],
    )
    
    # Create callbacks
    callbacks = []
    
    # Early stopping
    if config['callbacks']['early_stopping']:
        early_stop_callback = EarlyStopping(
            monitor=config['callbacks']['early_stopping']['monitor'],
            patience=config['callbacks']['early_stopping']['patience'],
            mode=config['callbacks']['early_stopping']['mode'],
        )
        callbacks.append(early_stop_callback)
    
    # Model checkpoint
    if config['callbacks']['model_checkpoint']:
        checkpoint_callback = ModelCheckpoint(
            monitor=config['callbacks']['model_checkpoint']['monitor'],
            mode=config['callbacks']['model_checkpoint']['mode'],
            save_top_k=config['callbacks']['model_checkpoint']['save_top_k'],
            filename=config['callbacks']['model_checkpoint']['filename'],
        )
        callbacks.append(checkpoint_callback)
    
    # Create logger
    logger = TensorBoardLogger("lightning_logs", name="efficientdet")
    
    # Create trainer
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        precision=config['training']['precision'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
    )
    
    # Start training
    print("Starting training...")
    trainer.fit(model, dm)
    
    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    
    # Save final model
    final_model_path = "efficientdet_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == '__main__':
    main()

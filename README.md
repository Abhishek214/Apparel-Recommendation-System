**Title:** Apparel Recommendation: Enhancing Shopping Experience with AI

**Introduction:**
With the ever-increasing popularity of online shopping, personalized product recommendations have become essential for enhancing the customer experience. In this project, we have developed an apparel recommendation engine that leverages the Amazon API, Natural Language Toolkit (NLTK), and Keras to extract apparel details and recommend similar products to users. By utilizing various techniques such as bag of words, TF-IDF, word2vec, and convolutional neural networks, we provide accurate and personalized recommendations to customers.

**Problem Statement:**
Our goal is to recommend similar apparel products based on a given product's details. We extract data from a JSON file containing over 180,000 apparel images from Amazon.com. The recommendations are generated based on fields such as ASIN, brand, color, product type, image URL, title, and price. We employ seven different approaches to ensure diverse and effective recommendations.

**Approaches Used:**
1. Bag of Words Model: Utilizes a bag of words representation to find similar products based on textual information.

2. TF-IDF Model: Implements the TF-IDF algorithm to measure the importance of words in the product descriptions and suggests similar items.

3. IDF Model: Focuses on the inverse document frequency (IDF) of words to identify related products.

4. Word2Vec Model: Employs word embeddings to capture semantic relationships between words and generate recommendations based on similarity.

5. IDF Weighted Word2Vec Model: Combines the IDF and Word2Vec models, weighting the word vectors based on their IDF scores for improved recommendation accuracy.

6. Weighted Similarity using Brand and Color: Uses the brand and color information to calculate weighted similarity scores and recommend relevant apparel products.

7. Visual Features Based on Convolutional Neural Networks: Extracts visual features from apparel images using a pre-trained CNN model and recommends visually similar items.

**Datasets and Inputs:**
- tops_fashion.json: JSON file containing the details of over 180,000 apparel images from Amazon.com.
- 16k_apparel_preprocessed pickle file: Preprocessed data of 16,000 apparel items for training and evaluation.
- Trained Word2Vec Model: Pre-trained Word2Vec model used for generating word embeddings.
- Trained CNN Model: Pre-trained Convolutional Neural Network (CNN) model for extracting visual features from apparel images.
- 16k_data_features_asins: ASINs (Amazon Standard Identification Numbers) of 16,000 apparel items.
- 16k_data_cnn_features.npy: Extracted CNN features for the 16,000 apparel items.

**Software Requirements:**
- Anaconda with additional packages: TensorFlow, Plotly, PIL.
- GPU for training the CNN and Word2Vec models.

**Execution and Running Code:**
1. Clone the repository by executing the following command in the terminal: `git clone https://github.com/Abhinav1004/Apparel-Recommendation.git`.
2. Launch Jupyter Notebook by executing the command: `jupyter notebook Apparel_Recommendation.ipynb`.
3. Run the notebook cells by pressing Shift+Enter to execute the code.
4. Observe the results, including the recommended apparel items and the evaluation metrics.

**Observations:**
During the project, we trained and evaluated seven different models for recommending similar apparel products. For each model, we recommended the top 20 apparel items with the least Euclidean distance. We calculated the average Euclidean distance for each model and compared their performance using line plots and bar graphs.

**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF

7.CNN
"""
Florence 2 Object Detection Fine-tuning for Document Layout Parsing
Optimized code for document layout parsing with class balancing, hyperparameter tuning,
and efficient multi-GPU training.
"""

import os
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional, Union
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
from functools import partial
import random
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed()

# Constants
MODEL_BASE = "microsoft/florence-2-base"
MODEL_LARGE = "microsoft/florence-2-large"
CLASS_NAMES = ['title', 'table', 'logo', 'signature', 'id', 'tableheader', 'stamp']
CLASS_WEIGHTS = [1.0, 1.25, 1.3, 3.4, 3.6, 5.2, 22.5]  # Inverse class frequency (normalized)

# Environment setup
def setup_environment():
    """Setup GPU environment and return device configuration"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA devices")
        for i in range(device_count):
            device_properties = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_properties.name}, "
                        f"Memory: {device_properties.total_memory / 1e9:.2f} GB")
        
        use_distributed = device_count > 1
        device = torch.device("cuda")
    else:
        logger.warning("No CUDA devices available, using CPU")
        use_distributed = False
        device = torch.device("cpu")
    
    return device, use_distributed

DEVICE, USE_DISTRIBUTED = setup_environment()

# Data augmentation
class DocumentLayoutAugmentation:
    """Advanced augmentation for document layout data with class-aware transforms"""
    
    def __init__(self, rare_class_augment_factor=2.0):
        self.rare_class_names = ['stamp', 'tableheader', 'id', 'signature']  # Classes with fewer samples
        self.rare_class_augment_factor = rare_class_augment_factor
        
        # Basic transforms for all images
        self.basic_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 0.5))], p=0.2),
        ])
        
        # Advanced transforms for rare classes
        self.rare_class_transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomApply([transforms.RandomRotation(5)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, scale=(0.95, 1.05))], p=0.5),
        ])
    
    def __call__(self, image, annotation):
        """Apply transformations based on classes present in the image"""
        # Check if image contains rare classes
        contains_rare_class = False
        if 'objects' in annotation:
            for obj in annotation['objects']:
                if obj['class'] in self.rare_class_names:
                    contains_rare_class = True
                    break
        
        # Apply appropriate transforms
        if contains_rare_class and random.random() < self.rare_class_augment_factor:
            return self.rare_class_transforms(image)
        else:
            return self.basic_transforms(image)

# Dataset Classes
class JSONLDataset:
    """Dataset for loading JSONL annotations with corresponding images"""
    
    def __init__(self, jsonl_file_path: str, image_directory_path: str):
        self.jsonl_file_path = jsonl_file_path
        self.image_directory_path = image_directory_path
        self.entries = self._load_entries()
        self.class_counts = self._count_classes()
        logger.info(f"Loaded {len(self.entries)} entries from {jsonl_file_path}")
        logger.info(f"Class distribution: {self.class_counts}")
    
    def _load_entries(self) -> List[Dict[str, Any]]:
        entries = []
        with open(self.jsonl_file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                entries.append(data)
        return entries
    
    def _count_classes(self) -> Dict[str, int]:
        """Count occurrences of each class in the dataset"""
        class_counts = {cls: 0 for cls in CLASS_NAMES}
        
        for entry in self.entries:
            if 'objects' in entry:
                for obj in entry['objects']:
                    if 'class' in obj and obj['class'] in class_counts:
                        class_counts[obj['class']] += 1
        
        return class_counts
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        if idx < 0 or idx >= len(self.entries):
            raise IndexError("Index out of range")
        
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return (image, entry)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file {image_path} not found.")
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a placeholder image in case of error
            placeholder = Image.new('RGB', (640, 640), color=(200, 200, 200))
            return (placeholder, entry)

class DetectionDataset(Dataset):
    """Dataset for object detection with balanced sampling weights"""
    
    def __init__(self, jsonl_file_path: str, image_directory_path: str, 
                transform=None, is_training=True):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)
        self.transform = transform
        self.is_training = is_training
        self.class_weights = self._compute_sample_weights()
    
    def _compute_sample_weights(self) -> List[float]:
        """Compute sample weights for balanced sampling"""
        weights = []
        
        for idx in range(len(self.dataset)):
            _, entry = self.dataset[idx]
            if 'objects' in entry and entry['objects']:
                # Count classes in this sample
                sample_classes = {}
                for obj in entry['objects']:
                    if 'class' in obj and obj['class'] in CLASS_NAMES:
                        cls_name = obj['class']
                        sample_classes[cls_name] = sample_classes.get(cls_name, 0) + 1
                
                # Calculate weight based on inverse frequency
                if sample_classes:
                    # Get index of each class present in this sample
                    class_indices = [CLASS_NAMES.index(cls) for cls in sample_classes.keys()]
                    # Use the maximum class weight for this sample
                    weight = max([CLASS_WEIGHTS[i] for i in class_indices])
                else:
                    weight = 1.0
            else:
                weight = 1.0
            
            weights.append(weight)
        
        return weights
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, data = self.dataset[idx]
        
        # Apply transforms if provided and in training mode
        if self.transform is not None and self.is_training:
            image = self.transform(image, data)
        
        # Get prefix and suffix from data
        prefix = data.get('prefix', '')
        suffix = data.get('suffix', '')
        
        return prefix, suffix, image

# Custom collate function with error handling
def collate_fn(batch, processor, device):
    questions, answers, images = zip(*batch)
    
    # Filter out invalid images
    valid_indices = []
    for i, img in enumerate(images):
        if img is not None:
            valid_indices.append(i)
    
    if not valid_indices:
        logger.warning("No valid images in batch!")
        # Return empty tensors
        empty_tensor = torch.zeros(1, 1).to(device)
        return {"input_ids": empty_tensor, "attention_mask": empty_tensor, 
                "pixel_values": empty_tensor}, [""]
    
    # Keep only valid items
    filtered_questions = [questions[i] for i in valid_indices]
    filtered_answers = [answers[i] for i in valid_indices]
    filtered_images = [images[i] for i in valid_indices]
    
    try:
        # Process inputs with error handling
        inputs = processor(
            text=filtered_questions, 
            images=filtered_images, 
            return_tensors="pt", 
            padding=True
        )
        
        # Check for invalid tensors
        if inputs["input_ids"].numel() == 0 or inputs["pixel_values"].numel() == 0:
            logger.warning("Empty tensor detected in processor output")
            # Return placeholders
            empty_tensor = torch.zeros(1, 1).to(device)
            return {"input_ids": empty_tensor, "attention_mask": empty_tensor, 
                    "pixel_values": empty_tensor}, [""]
        
        # Move tensors to device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        
        return inputs, filtered_answers
    
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        # Return empty placeholders
        empty_tensor = torch.zeros(1, 1).to(device)
        return {"input_ids": empty_tensor, "attention_mask": empty_tensor, 
                "pixel_values": empty_tensor}, [""]

# Focal Loss implementation for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Hyperparameter tuning with Optuna
def optuna_hyperparameter_search(train_dataset, val_dataset, processor, n_trials=10):
    """Hyperparameter tuning with Optuna"""
    logger.info("Starting hyperparameter tuning with Optuna")
    
    def objective(trial):
        """Objective function for Optuna optimization"""
        # Sample hyperparameters
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        lr = trial.suggest_float('lr', 1e-6, 2e-5, log=True)
        lora_r = trial.suggest_categorical('lora_r', [8, 16, 32])
        lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64])
        lora_dropout = trial.suggest_float('lora_dropout', 0.05, 0.2)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.05, 0.2)
        weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
        
        logger.info(f"Trial with params: batch_size={batch_size}, lr={lr}, "
                   f"lora_r={lora_r}, lora_alpha={lora_alpha}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
            num_workers=2, shuffle=True, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            collate_fn=lambda batch: collate_fn(batch, processor, DEVICE),
            num_workers=2, pin_memory=True
        )
        
        # Initialize model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_BASE, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(DEVICE)
        
        # LoRA config
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "lm_head"],
            task_type="CAUSAL_LM",
            lora_dropout=lora_dropout,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        
        # Create PEFT model
        peft_model = get_peft_model(model, lora_config)
        
        # Optimizer and scheduler
        optimizer = AdamW(
            peft_model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Set up num training steps
        num_train_steps = len(train_loader) * 3  # 3 epochs for tuning
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        
        scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
        
        # Quick training for 3 epochs
        best_val_loss = float('inf')
        
        for epoch in range(3):
            # Training
            peft_model.train()
            train_loss = 0
            for batch_idx, (inputs, answers) in enumerate(train_loader):
                # Skip problematic batches
                if any(tensor.numel() == 0 for tensor in inputs.values() if isinstance(tensor, torch.Tensor)):
                    continue
                
                # Prepare labels
                labels = processor.tokenizer(
                    text=answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).input_ids.to(DEVICE)
                
                # Forward pass
                outputs = peft_model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    labels=labels
                )
                
                loss = outputs.loss
                if loss.dim() > 0:
                    loss = loss.mean()
                
                # Check for NaN
                if torch.isnan(loss):
                    continue
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
            
            # Validation
            peft_model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, answers) in enumerate(val_loader):
                    # Skip problematic batches
                    if any(tensor.numel() == 0 for tensor in inputs.values() if isinstance(tensor, torch.Tensor)):
                        continue
                    
                    # Prepare labels
                    labels = processor.tokenizer(
                        text=answers,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).input_ids.to(DEVICE)
                    
                    # Forward pass
                    outputs = peft_model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    if loss.dim() > 0:
                        loss = loss.mean()
                    
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        val_steps += 1
            
            # Calculate average validation loss
            avg_val_loss = val_loss / max(val_steps, 1)
            logger.info(f"Trial {trial.number}, Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")
            best_val_loss = min(best_val_loss, avg_val_loss)
            
            # Free memory
            torch.cuda.empty_cache()
        
        # Clean up
        del peft_model, model, optimizer, scheduler
        torch.cuda.empty_cache()
        
        return best_val_loss
    
    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # Log results
    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info("  Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")
    
    return study.best_trial.params

# Visualization functions
def plot_bboxes_on_image(image, bboxes, labels, save_path=None):
    """Plot bounding boxes on image with labels"""
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    
    ax = plt.gca()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(CLASS_NAMES)))
    color_map = {cls: colors[i] for i, cls in enumerate(CLASS_NAMES)}
    
    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        
        # Get color for this class
        color = color_map.get(label, 'red')
        
        rect = plt.Rectangle((xmin, ymin), width, height, fill=False, 
                           edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 2, label, 
               bbox=dict(facecolor=color, alpha=0.5), 
               fontsize=12, color='white')
    
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()

def plot_training_curves(train_losses, val_losses, save_path="training_curves.png"):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(dataset, title="Class Distribution"):
    """Plot class distribution from dataset"""
    class_counts = dataset.dataset.class_counts
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values())
    
    # Add count labels on top of bars
    for bar in bars:
      

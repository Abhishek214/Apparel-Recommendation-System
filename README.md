
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF



import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def collate_fn(batch):
    """
    Fixed collate function that properly handles image conversion for Florence-2 processor.
    This specifically addresses the 'Numpy is not available' error.
    """
    # Unpack the batch
    questions, answers, images = zip(*batch)
    
    # Convert all images to PIL Image format first
    pil_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL Image
            # First, ensure tensor is on CPU and in the right format
            img_cpu = img.cpu()
            
            # If the tensor is normalized, denormalize it
            if img_cpu.min() < 0:
                # Denormalize from ImageNet normalization
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                img_cpu = img_cpu * std + mean
                img_cpu = torch.clamp(img_cpu, 0, 1)
            
            # Convert to 0-255 range if needed
            if img_cpu.max() <= 1.0:
                img_cpu = (img_cpu * 255).to(torch.uint8)
            
            # Convert to PIL Image
            if img_cpu.dim() == 3:
                # [C, H, W] -> [H, W, C]
                img_np = img_cpu.permute(1, 2, 0).numpy()
                pil_img = Image.fromarray(img_np)
            else:
                # Skip invalid tensors
                pil_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        elif isinstance(img, Image.Image):
            # Already a PIL Image
            pil_img = img
        elif isinstance(img, np.ndarray):
            # Convert numpy array to PIL Image
            if img.shape[0] == 3:  # [C, H, W]
                img = np.transpose(img, (1, 2, 0))  # -> [H, W, C]
            pil_img = Image.fromarray(img.astype('uint8'))
        else:
            # Create a placeholder image
            pil_img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        # Ensure RGB mode
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        pil_images.append(pil_img)
    
    try:
        # Process all images and text together
        # IMPORTANT: The processor expects PIL Images, not numpy arrays or tensors
        inputs = processor(
            text=list(questions),
            images=pil_images,  # Pass PIL Images directly
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        return inputs, list(answers)
    
    except Exception as e:
        print(f"Error in collate_fn: {str(e)}")
        print(f"Batch size: {len(batch)}")
        print(f"Image types: {[type(img) for img in images]}")
        
        # Create a minimal valid batch as fallback
        # Use single sample to avoid batching issues
        fallback_question = questions[0] if questions else "What is this?"
        fallback_answer = answers[0] if answers else "This is an image."
        fallback_image = pil_images[0] if pil_images else Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        inputs = processor(
            text=[fallback_question],
            images=[fallback_image],
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        return inputs, [fallback_answer]

# ALTERNATIVE SOLUTION: Modify DetectionDataset to always return PIL Images
class DetectionDataset(Dataset):
    def __init__(self, jsonl_file_path: str, image_directory_path: str, transform=None, is_training=True):
        self.dataset = JSONLDataset(jsonl_file_path, image_directory_path)
        self.transform = transform
        self.is_training = is_training
        
        # Class weights for handling imbalance
        self.class_weights = self._calculate_class_weights()
        self.sample_weights = self._assign_sample_weights()
    
    def _calculate_class_weights(self):
        """Calculate weights for each class based on inverse frequency"""
        class_counts = self.dataset.class_counts
        total_samples = sum(class_counts.values())
        
        # Using inverse frequency weighting
        class_weights = {}
        for class_name, count in class_counts.items():
            class_weights[class_name] = total_samples / (len(class_counts) * count)
        
        return class_weights
    
    def _assign_sample_weights(self):
        """Assign weights to each sample based on its class"""
        weights = []
        for idx in range(len(self.dataset)):
            _, data = self.dataset[idx]
            class_name = self.dataset._extract_class_from_entry(data)
            weights.append(self.class_weights.get(class_name, 1.0))
        
        return weights
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            image, data = self.dataset[idx]
            prefix = data['prefix']
            suffix = data['suffix']
            
            # Ensure image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transformations if available
            if self.transform and self.is_training:
                # Convert PIL Image to numpy array for albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                
                # Check if transform returns a tensor or numpy array
                if 'image' in transformed:
                    transformed_image = transformed['image']
                    
                    # Convert tensor back to PIL Image for consistency
                    if isinstance(transformed_image, torch.Tensor):
                        # Denormalize if necessary
                        if transformed_image.min() < 0:
                            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
                            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
                            transformed_image = transformed_image * std + mean
                            transformed_image = torch.clamp(transformed_image, 0, 1)
                        
                        # Convert to 0-255 range
                        transformed_image = (transformed_image * 255).to(torch.uint8)
                        # [C, H, W] -> [H, W, C]
                        img_np = transformed_image.permute(1, 2, 0).cpu().numpy()
                        image = Image.fromarray(img_np)
                    else:
                        # Numpy array
                        image = Image.fromarray(transformed_image.astype('uint8'))
            
            # Return PIL Image instead of tensor
            # This ensures the Florence-2 processor gets the format it expects
            return prefix, suffix, image
            
        except Exception as e:
            print(f"Error loading sample at index {idx}: {str(e)}")
            # Return a placeholder sample
            placeholder_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
            return "What is this?", "This is a placeholder.", placeholder_image

# SIMPLIFIED AUGMENTATION WITHOUT NORMALIZATION
# Since we're returning PIL Images, we'll let the processor handle normalization
train_transform = A.Compose([
    # Document-specific augmentations
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=2, 
                       border_mode=0, p=0.7),
    
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3), p=0.8),
        A.MotionBlur(blur_limit=(3, 5), p=0.8),
    ], p=0.5),
    
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
    A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
    
    A.ToGray(p=0.2),
    
    # Remove normalization and ToTensorV2 since we're returning PIL Images
    # The processor will handle normalization
])

# No transforms for validation - just return PIL Images
val_transform = None  # The processor will handle any necessary preprocessing

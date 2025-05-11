
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF


def collate_fn(batch):
    """
    Custom collate function for Florence 2 model that properly handles image processing and tensor conversion.
    
    Args:
        batch: List of (question, answer, image) tuples from the dataset
        
    Returns:
        Tuple of (inputs_dict, answers) where inputs_dict contains model inputs and answers contains target texts
    """
    # Unpack the batch
    questions, answers, images = zip(*batch)
    
    # Process questions and images in batches rather than individually
    # This avoids the "Unable to create tensor" error
    processed_images = []
    for img in images:
        # Handle different image types
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:  # If it's already a tensor with shape [C, H, W]
                if img.shape[0] == 3:  # RGB image
                    processed_images.append(img.numpy())
                else:  # Ensure RGB
                    pil_img = transforms.ToPILImage()(img).convert('RGB')
                    processed_images.append(np.array(pil_img))
            else:
                # Skip invalid tensors
                print("Warning: Skipping invalid tensor in batch")
                # Add a placeholder to maintain batch size
                placeholder = np.zeros((3, 224, 224), dtype=np.float32)
                processed_images.append(placeholder)
        else:
            # Convert PIL image to numpy array
            if img.mode != 'RGB':
                img = img.convert('RGB')
            processed_images.append(np.array(img))
    
    try:
        # Process all images and text together
        inputs = processor(
            text=list(questions),
            images=processed_images,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        return inputs, list(answers)
    
    except Exception as e:
        print(f"Error in collate_fn: {str(e)}")
        print(f"Batch size: {len(batch)}")
        print(f"First image type: {type(images[0])}")
        if hasattr(images[0], 'size'):
            print(f"First image size: {images[0].size}")
        elif isinstance(images[0], torch.Tensor):
            print(f"First tensor shape: {images[0].shape}")
        
        # Create a minimal valid batch as fallback
        empty_inputs = {
            "input_ids": torch.zeros((1, 1), dtype=torch.long).to(DEVICE),
            "attention_mask": torch.zeros((1, 1), dtype=torch.long).to(DEVICE),
            "pixel_values": torch.zeros((1, 3, 224, 224), dtype=torch.float).to(DEVICE),
        }
        return empty_inputs, answers[0:1]



-----------

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
            # Weight = total_samples / (num_classes * count)
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
                # With ToTensorV2() in the transform pipeline, this returns a tensor
                # Otherwise, we'll get a numpy array
                if 'image' in transformed:
                    if isinstance(transformed['image'], torch.Tensor):
                        return prefix, suffix, transformed['image']
                    else:
                        # Convert numpy to tensor if not done by transform
                        image_tensor = torch.from_numpy(transformed['image'].transpose(2, 0, 1)).float() / 255.0
                        return prefix, suffix, image_tensor
            
            # For validation or if no transform is applied
            # Convert PIL to tensor manually
            image_tensor = transforms.ToTensor()(image)
            return prefix, suffix, image_tensor
            
        except Exception as e:
            print(f"Error loading sample at index {idx}: {str(e)}")
            # Return a placeholder sample to avoid breaking the dataloader
            # Create a black image tensor
            placeholder_image = torch.zeros((3, 224, 224), dtype=torch.float32)
            return "What is this?", "This is a placeholder.", placeholder_image



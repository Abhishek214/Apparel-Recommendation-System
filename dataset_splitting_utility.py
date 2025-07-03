#!/usr/bin/env python3
"""
COCO Dataset Splitting Utility
Split COCO format dataset into train/validation sets with proper stratification

Usage:
    python split_coco_dataset.py --input dataset.json --output_dir ./data --train_ratio 0.8
"""
import os
import json
import argparse
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import random
import numpy as np

class COCODatasetSplitter:
    """Split COCO dataset into train/validation sets with stratification"""
    
    def __init__(self, input_path, output_dir, train_ratio=0.8, random_seed=42):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = 1.0 - train_ratio
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_images_dir = self.output_dir / 'train_images'
        self.val_images_dir = self.output_dir / 'val_images'
        self.train_images_dir.mkdir(exist_ok=True)
        self.val_images_dir.mkdir(exist_ok=True)
        
        # Load dataset
        self.load_dataset()
        
    def load_dataset(self):
        """Load COCO dataset from JSON file"""
        print(f"Loading dataset from: {self.input_path}")
        
        with open(self.input_path, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)
        
        print(f"Dataset loaded successfully")
        print(f"Images: {len(self.coco_data['images'])}")
        print(f"Annotations: {len(self.coco_data['annotations'])}")
        print(f"Categories: {len(self.coco_data['categories'])}")
        
        # Create mappings for easier access
        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.annotations_by_image = defaultdict(list)
        
        for ann in self.coco_data['annotations']:
            self.annotations_by_image[ann['image_id']].append(ann)
        
        print(f"Images with annotations: {len(self.annotations_by_image)}")
    
    def analyze_dataset(self):
        """Analyze dataset distribution"""
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        # Category distribution
        category_counts = Counter()
        for ann in self.coco_data['annotations']:
            category_counts[ann['category_id']] += 1
        
        # Create category name mapping
        cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        print("\nCategory Distribution:")
        for cat_id, count in category_counts.most_common():
            cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            percentage = (count / len(self.coco_data['annotations'])) * 100
            print(f"  {cat_name}: {count} annotations ({percentage:.1f}%)")
        
        # Images per category (images that contain each category)
        images_per_category = defaultdict(set)
        for ann in self.coco_data['annotations']:
            images_per_category[ann['category_id']].add(ann['image_id'])
        
        print("\nImages per Category:")
        for cat_id in sorted(images_per_category.keys()):
            cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
            img_count = len(images_per_category[cat_id])
            percentage = (img_count / len(self.coco_data['images'])) * 100
            print(f"  {cat_name}: {img_count} images ({percentage:.1f}%)")
        
        # Annotations per image statistics
        anns_per_image = [len(anns) for anns in self.annotations_by_image.values()]
        if anns_per_image:
            print(f"\nAnnotations per Image:")
            print(f"  Mean: {np.mean(anns_per_image):.2f}")
            print(f"  Median: {np.median(anns_per_image):.2f}")
            print(f"  Min: {min(anns_per_image)}")
            print(f"  Max: {max(anns_per_image)}")
        
        return category_counts, images_per_category
    
    def stratified_split(self):
        """Perform stratified split ensuring balanced distribution"""
        print(f"\n" + "="*50)
        print("PERFORMING STRATIFIED SPLIT")
        print("="*50)
        
        # Get images with annotations only
        image_ids_with_anns = list(self.annotations_by_image.keys())
        
        # Group images by their category combinations
        image_category_signatures = {}
        for img_id in image_ids_with_anns:
            # Get unique categories in this image
            categories = set()
            for ann in self.annotations_by_image[img_id]:
                categories.add(ann['category_id'])
            
            # Create signature (sorted tuple of category IDs)
            signature = tuple(sorted(categories))
            image_category_signatures[img_id] = signature
        
        # Group images by signature
        signature_to_images = defaultdict(list)
        for img_id, signature in image_category_signatures.items():
            signature_to_images[signature].append(img_id)
        
        print(f"Found {len(signature_to_images)} unique category combinations:")
        cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        for signature, img_list in signature_to_images.items():
            cat_names = [cat_id_to_name.get(cat_id, f"Unknown_{cat_id}") for cat_id in signature]
            print(f"  {'+'.join(cat_names)}: {len(img_list)} images")
        
        # Split each group proportionally
        train_images = []
        val_images = []
        
        for signature, img_list in signature_to_images.items():
            # Shuffle the list
            random.shuffle(img_list)
            
            # Calculate split point
            n_train = int(len(img_list) * self.train_ratio)
            
            # Ensure at least one image in each split if possible
            if len(img_list) >= 2:
                n_train = max(1, min(n_train, len(img_list) - 1))
            
            # Split
            train_images.extend(img_list[:n_train])
            val_images.extend(img_list[n_train:])
        
        print(f"\nSplit Results:")
        print(f"  Training images: {len(train_images)} ({len(train_images)/len(image_ids_with_anns)*100:.1f}%)")
        print(f"  Validation images: {len(val_images)} ({len(val_images)/len(image_ids_with_anns)*100:.1f}%)")
        
        return train_images, val_images
    
    def create_split_datasets(self, train_image_ids, val_image_ids):
        """Create separate COCO files for train and validation"""
        print(f"\n" + "="*50)
        print("CREATING SPLIT DATASETS")
        print("="*50)
        
        # Create train dataset
        train_data = {
            'info': self.coco_data['info'].copy(),
            'licenses': self.coco_data['licenses'].copy(),
            'categories': self.coco_data['categories'].copy(),
            'images': [],
            'annotations': []
        }
        
        # Create val dataset
        val_data = {
            'info': self.coco_data['info'].copy(),
            'licenses': self.coco_data['licenses'].copy(),
            'categories': self.coco_data['categories'].copy(),
            'images': [],
            'annotations': []
        }
        
        # Add training images and annotations
        train_ann_id = 1
        for img_id in train_image_ids:
            # Add image info
            img_info = self.image_id_to_info[img_id].copy()
            train_data['images'].append(img_info)
            
            # Add annotations
            for ann in self.annotations_by_image[img_id]:
                ann_copy = ann.copy()
                ann_copy['id'] = train_ann_id
                train_data['annotations'].append(ann_copy)
                train_ann_id += 1
        
        # Add validation images and annotations
        val_ann_id = 1
        for img_id in val_image_ids:
            # Add image info
            img_info = self.image_id_to_info[img_id].copy()
            val_data['images'].append(img_info)
            
            # Add annotations
            for ann in self.annotations_by_image[img_id]:
                ann_copy = ann.copy()
                ann_copy['id'] = val_ann_id
                val_data['annotations'].append(ann_copy)
                val_ann_id += 1
        
        # Save datasets
        train_json_path = self.output_dir / 'train_coco.json'
        val_json_path = self.output_dir / 'val_coco.json'
        
        with open(train_json_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training dataset saved: {train_json_path}")
        print(f"Validation dataset saved: {val_json_path}")
        
        # Print final statistics
        self.print_split_statistics(train_data, val_data)
        
        return train_data, val_data
    
    def copy_images(self, train_image_ids, val_image_ids, source_image_dir):
        """Copy images to train/val directories"""
        if not source_image_dir:
            print("No source image directory provided - skipping image copying")
            return
        
        source_dir = Path(source_image_dir)
        if not source_dir.exists():
            print(f"Source image directory does not exist: {source_dir}")
            return
        
        print(f"\n" + "="*50)
        print("COPYING IMAGES")
        print("="*50)
        
        # Copy training images
        print(f"Copying {len(train_image_ids)} training images...")
        for img_id in train_image_ids:
            img_info = self.image_id_to_info[img_id]
            src_path = source_dir / img_info['file_name']
            dst_path = self.train_images_dir / img_info['file_name']
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
        
        # Copy validation images
        print(f"Copying {len(val_image_ids)} validation images...")
        for img_id in val_image_ids:
            img_info = self.image_id_to_info[img_id]
            src_path = source_dir / img_info['file_name']
            dst_path = self.val_images_dir / img_info['file_name']
            
            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Warning: Image not found: {src_path}")
        
        print(f"Images copied to:")
        print(f"  Training: {self.train_images_dir}")
        print(f"  Validation: {self.val_images_dir}")
    
    def print_split_statistics(self, train_data, val_data):
        """Print detailed statistics about the split"""
        print(f"\n" + "="*50)
        print("SPLIT STATISTICS")
        print("="*50)
        
        # Overall statistics
        total_images = len(train_data['images']) + len(val_data['images'])
        total_annotations = len(train_data['annotations']) + len(val_data['annotations'])
        
        print(f"Total Images: {total_images}")
        print(f"  Training: {len(train_data['images'])} ({len(train_data['images'])/total_images*100:.1f}%)")
        print(f"  Validation: {len(val_data['images'])} ({len(val_data['images'])/total_images*100:.1f}%)")
        
        print(f"\nTotal Annotations: {total_annotations}")
        print(f"  Training: {len(train_data['annotations'])} ({len(train_data['annotations'])/total_annotations*100:.1f}%)")
        print(f"  Validation: {len(val_data['annotations'])} ({len(val_data['annotations'])/total_annotations*100:.1f}%)")
        
        # Category distribution
        cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        train_category_counts = Counter(ann['category_id'] for ann in train_data['annotations'])
        val_category_counts = Counter(ann['category_id'] for ann in val_data['annotations'])
        
        print(f"\nCategory Distribution:")
        print(f"{'Category':<15} {'Train':<8} {'Val':<8} {'Train%':<8} {'Val%':<8}")
        print("-" * 55)
        
        for cat_id in sorted(cat_id_to_name.keys()):
            cat_name = cat_id_to_name[cat_id]
            train_count = train_category_counts.get(cat_id, 0)
            val_count = val_category_counts.get(cat_id, 0)
            total_cat = train_count + val_count
            
            if total_cat > 0:
                train_pct = train_count / total_cat * 100
                val_pct = val_count / total_cat * 100
                print(f"{cat_name:<15} {train_count:<8} {val_count:<8} {train_pct:<8.1f} {val_pct:<8.1f}")
    
    def split_dataset(self, source_image_dir=None):
        """Main method to split the dataset"""
        # Analyze original dataset
        self.analyze_dataset()
        
        # Perform stratified split
        train_image_ids, val_image_ids = self.stratified_split()
        
        # Create split datasets
        train_data, val_data = self.create_split_datasets(train_image_ids, val_image_ids)
        
        # Copy images if source directory provided
        if source_image_dir:
            self.copy_images(train_image_ids, val_image_ids, source_image_dir)
        
        print(f"\n" + "="*50)
        print("DATASET SPLIT COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        print(f"\nOutput files:")
        print(f"  Training annotations: {self.output_dir / 'train_coco.json'}")
        print(f"  Validation annotations: {self.output_dir / 'val_coco.json'}")
        if source_image_dir:
            print(f"  Training images: {self.train_images_dir}")
            print(f"  Validation images: {self.val_images_dir}")
        
        print(f"\nUse these paths in your EfficientDet config:")
        print(f"data:")
        print(f"  train_annotations: '{self.output_dir / 'train_coco.json'}'")
        print(f"  val_annotations: '{self.output_dir / 'val_coco.json'}'")
        if source_image_dir:
            print(f"  train_images: '{self.train_images_dir}'")
            print(f"  val_images: '{self.val_images_dir}'")


def main():
    parser = argparse.ArgumentParser(description='Split COCO dataset into train/validation sets')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input COCO JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for split datasets')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of training data (default: 0.8)')
    parser.add_argument('--source_images', type=str,
                       help='Source directory containing images (optional - for copying images to split directories)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible splits (default: 42)')
    parser.add_argument('--min_val_samples', type=int, default=1,
                       help='Minimum samples per category in validation set (default: 1)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.1 <= args.train_ratio <= 0.9):
        print("Error: train_ratio must be between 0.1 and 0.9")
        return
    
    if not Path(args.input).exists():
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    if args.source_images and not Path(args.source_images).exists():
        print(f"Error: Source images directory does not exist: {args.source_images}")
        return
    
    # Create splitter and split dataset
    splitter = COCODatasetSplitter(
        input_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )
    
    splitter.split_dataset(source_image_dir=args.source_images)


if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        print("COCO Dataset Splitter")
        print("====================")
        print("\nExample usage:")
        print("python split_coco_dataset.py --input dataset.json --output_dir ./data --train_ratio 0.8")
        print("\nWith image copying:")
        print("python split_coco_dataset.py --input dataset.json --output_dir ./data --train_ratio 0.8 --source_images ./images")
        print("\nArguments:")
        print("  --input: Path to COCO JSON file")
        print("  --output_dir: Directory to create train/val splits")
        print("  --train_ratio: Fraction for training (0.1-0.9, default: 0.8)")
        print("  --source_images: Source image directory (optional)")
        print("  --random_seed: Random seed for reproducibility (default: 42)")
    else:
        main()
import json
import os
import glob
from datetime import datetime
import cv2
import numpy as np
from typing import List, Dict, Tuple
import argparse

class LabelMeToCOCOConverter:
    """
    Convert LabelMe format annotations to COCO format for EfficientDet training.
    
    LabelMe format: One JSON file per image with polygon annotations
    COCO format: Single JSON file with all annotations and metadata
    """
    
    def __init__(self):
        self.coco_data = {
            "info": {
                "description": "Custom dataset for EfficientDet training",
                "url": "",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Custom Dataset",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Category mapping for your classes
        self.categories = {
            "signature": 1,
            "barcode": 2,
            "chop": 3,
            "stamp": 4
        }
        
        self.image_id = 1
        self.annotation_id = 1
        
        # Initialize categories in COCO format
        for name, cat_id in self.categories.items():
            self.coco_data["categories"].append({
                "id": cat_id,
                "name": name,
                "supercategory": "object"
            })
    
    def polygon_to_bbox(self, points: List[List[float]]) -> Tuple[float, float, float, float]:
        """
        Convert polygon points to bounding box format [x, y, width, height].
        
        Args:
            points: List of [x, y] coordinates
            
        Returns:
            Tuple of (x_min, y_min, width, height)
        """
        if not points:
            return (0, 0, 0, 0)
        
        # Extract x and y coordinates
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]
        
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        width = x_max - x_min
        height = y_max - y_min
        
        return (x_min, y_min, width, height)
    
    def calculate_polygon_area(self, points: List[List[float]]) -> float:
        """
        Calculate area of polygon using shoelace formula.
        
        Args:
            points: List of [x, y] coordinates
            
        Returns:
            Area of the polygon
        """
        if len(points) < 3:
            return 0.0
        
        # Shoelace formula
        area = 0.0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """
        Get image dimensions from file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (width, height)
        """
        try:
            img = cv2.imread(image_path)
            if img is not None:
                height, width = img.shape[:2]
                return width, height
            else:
                print(f"Warning: Could not read image {image_path}")
                return 0, 0
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return 0, 0
    
    def process_labelme_json(self, json_path: str, image_dir: str) -> bool:
        """
        Process a single LabelMe JSON file and add to COCO dataset.
        
        Args:
            json_path: Path to LabelMe JSON file
            image_dir: Directory containing images
            
        Returns:
            True if processed successfully, False otherwise
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            # Get image information
            image_filename = labelme_data.get('imagePath', '')
            if not image_filename:
                print(f"Warning: No imagePath in {json_path}")
                return False
            
            # Handle relative paths
            image_path = os.path.join(image_dir, os.path.basename(image_filename))
            if not os.path.exists(image_path):
                # Try looking for the image with the same base name as JSON
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    potential_path = os.path.join(image_dir, base_name + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        image_filename = os.path.basename(potential_path)
                        break
                else:
                    print(f"Warning: Image not found for {json_path}")
                    return False
            
            # Get image dimensions
            if 'imageHeight' in labelme_data and 'imageWidth' in labelme_data:
                width = labelme_data['imageWidth']
                height = labelme_data['imageHeight']
            else:
                width, height = self.get_image_dimensions(image_path)
                if width == 0 or height == 0:
                    return False
            
            # Add image to COCO dataset
            image_info = {
                "id": self.image_id,
                "width": width,
                "height": height,
                "file_name": image_filename,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": datetime.now().isoformat()
            }
            self.coco_data["images"].append(image_info)
            
            # Process shapes/annotations
            shapes = labelme_data.get('shapes', [])
            
            for shape in shapes:
                label = shape.get('label', '').lower()
                
                # Map label to category ID
                category_id = None
                for category_name, cat_id in self.categories.items():
                    if category_name.lower() in label or label in category_name.lower():
                        category_id = cat_id
                        break
                
                if category_id is None:
                    print(f"Warning: Unknown label '{label}' in {json_path}")
                    continue
                
                points = shape.get('points', [])
                if len(points) < 3:  # Need at least 3 points for a polygon
                    print(f"Warning: Insufficient points for shape in {json_path}")
                    continue
                
                # Convert polygon to bounding box
                bbox = self.polygon_to_bbox(points)
                area = self.calculate_polygon_area(points)
                
                # Flatten points for COCO segmentation format
                segmentation = []
                for point in points:
                    segmentation.extend([float(point[0]), float(point[1])])
                
                # Create annotation
                annotation = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                
                self.coco_data["annotations"].append(annotation)
                self.annotation_id += 1
            
            self.image_id += 1
            return True
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            return False
    
    def convert_dataset(self, labelme_dir: str, image_dir: str, output_path: str):
        """
        Convert entire dataset from LabelMe to COCO format.
        
        Args:
            labelme_dir: Directory containing LabelMe JSON files
            image_dir: Directory containing images
            output_path: Path for output COCO JSON file
        """
        # Find all JSON files
        json_files = glob.glob(os.path.join(labelme_dir, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {labelme_dir}")
            return
        
        print(f"Found {len(json_files)} JSON files to process")
        
        processed_count = 0
        
        for json_file in json_files:
            print(f"Processing: {os.path.basename(json_file)}")
            
            if self.process_labelme_json(json_file, image_dir):
                processed_count += 1
            else:
                print(f"Failed to process: {json_file}")
        
        print(f"\nProcessed {processed_count}/{len(json_files)} files successfully")
        print(f"Total images: {len(self.coco_data['images'])}")
        print(f"Total annotations: {len(self.coco_data['annotations'])}")
        
        # Count annotations per category
        category_counts = {}
        for ann in self.coco_data['annotations']:
            cat_id = ann['category_id']
            cat_name = next(cat['name'] for cat in self.coco_data['categories'] if cat['id'] == cat_id)
            category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
        
        print("\nAnnotations per category:")
        for cat_name, count in category_counts.items():
            print(f"  {cat_name}: {count}")
        
        # Save COCO dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nCOCO dataset saved to: {output_path}")
    
    def validate_coco_dataset(self, coco_path: str):
        """
        Validate the generated COCO dataset.
        
        Args:
            coco_path: Path to COCO JSON file
        """
        try:
            with open(coco_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
            
            # Basic validation
            required_keys = ['images', 'annotations', 'categories']
            for key in required_keys:
                if key not in coco_data:
                    print(f"Error: Missing required key '{key}' in COCO dataset")
                    return False
            
            # Check if we have data
            if not coco_data['images']:
                print("Error: No images in dataset")
                return False
            
            if not coco_data['annotations']:
                print("Error: No annotations in dataset")
                return False
            
            if not coco_data['categories']:
                print("Error: No categories in dataset")
                return False
            
            # Validate references
            image_ids = set(img['id'] for img in coco_data['images'])
            category_ids = set(cat['id'] for cat in coco_data['categories'])
            
            invalid_refs = 0
            for ann in coco_data['annotations']:
                if ann['image_id'] not in image_ids:
                    invalid_refs += 1
                if ann['category_id'] not in category_ids:
                    invalid_refs += 1
            
            if invalid_refs > 0:
                print(f"Warning: Found {invalid_refs} invalid references in annotations")
            
            print("COCO dataset validation completed successfully!")
            return True
            
        except Exception as e:
            print(f"Error validating COCO dataset: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Convert LabelMe format to COCO format for EfficientDet')
    parser.add_argument('--labelme_dir', type=str, required=True,
                       help='Directory containing LabelMe JSON files')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for COCO JSON file')
    parser.add_argument('--validate', action='store_true',
                       help='Validate the generated COCO dataset')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.labelme_dir):
        print(f"Error: LabelMe directory does not exist: {args.labelme_dir}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory does not exist: {args.image_dir}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert dataset
    converter = LabelMeToCOCOConverter()
    converter.convert_dataset(args.labelme_dir, args.image_dir, args.output)
    
    # Validate if requested
    if args.validate:
        print("\nValidating generated COCO dataset...")
        converter.validate_coco_dataset(args.output)


if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python convert_labelme_to_coco.py --labelme_dir ./annotations --image_dir ./images --output ./coco_dataset.json --validate")
        print("\nFor your specific use case:")
        print("python convert_labelme_to_coco.py --labelme_dir /path/to/your/json/files --image_dir /path/to/your/images --output ./efficientdet_dataset.json --validate")
    else:
        main()

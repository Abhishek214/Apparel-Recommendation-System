import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, input_size=640):
    """
    Preprocess image for EfficientDet ONNX model inference
    
    Args:
        image_path (str): Path to input image
        input_size (int): Model input size (640 for D1)
    
    Returns:
        numpy.ndarray: Preprocessed image array (1, 3, 640, 640)
        dict: Metadata for post-processing (original size, scale factor)
    """
    
    # Method 1: Using OpenCV (recommended for documents)
    def preprocess_with_opencv(image_path, input_size):
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        original_height, original_width = img.shape[:2]
        
        # Calculate scale factor to maintain aspect ratio
        scale = min(input_size / original_width, input_size / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image maintaining aspect ratio
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image (pad to square)
        padded_img = np.ones((input_size, input_size, 3), dtype=np.uint8) * 114  # Gray padding
        
        # Calculate padding offsets to center the image
        pad_x = (input_size - new_width) // 2
        pad_y = (input_size - new_height) // 2
        
        # Place resized image in center of padded image
        padded_img[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = img_resized
        
        # Normalize using ImageNet statistics (same as training)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Convert to float and normalize to [0, 1]
        img_normalized = padded_img.astype(np.float32) / 255.0
        
        # Apply normalization
        img_normalized = (img_normalized - mean) / std
        
        # Convert from HWC to CHW format
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # Metadata for post-processing
        metadata = {
            'original_width': original_width,
            'original_height': original_height,
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'input_size': input_size
        }
        
        return img_batch, metadata
    
    # Method 2: Using PIL (alternative)
    def preprocess_with_pil(image_path, input_size):
        # Read image
        img = Image.open(image_path).convert('RGB')
        original_width, original_height = img.size
        
        # Calculate scale factor
        scale = min(input_size / original_width, input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Create padded image
        padded_img = Image.new('RGB', (input_size, input_size), (114, 114, 114))
        
        # Paste resized image in center
        pad_x = (input_size - new_width) // 2
        pad_y = (input_size - new_height) // 2
        padded_img.paste(img_resized, (pad_x, pad_y))
        
        # Convert to numpy array
        img_array = np.array(padded_img, dtype=np.float32)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
        
        img_normalized = (img_array - mean) / std
        
        # Convert to CHW and add batch dimension
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        metadata = {
            'original_width': original_width,
            'original_height': original_height,
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y,
            'input_size': input_size
        }
        
        return img_batch, metadata
    
    # Use OpenCV method (better for document images)
    return preprocess_with_opencv(image_path, input_size)

# Inverse preprocessing function for visualization
def postprocess_coordinates(detections, metadata):
    """
    Convert detection coordinates back to original image coordinates
    
    Args:
        detections: Detection results from model
        metadata: Metadata from preprocessing
    
    Returns:
        Adjusted detections in original image coordinates
    """
    
    scale = metadata['scale']
    pad_x = metadata['pad_x']
    pad_y = metadata['pad_y']
    
    adjusted_detections = []
    
    for detection in detections:
        # detection format: [x1, y1, x2, y2, confidence, class_id]
        x1, y1, x2, y2 = detection[:4]
        
        # Remove padding offset
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale
        
        # Clip to original image bounds
        x1 = max(0, min(x1, metadata['original_width']))
        y1 = max(0, min(y1, metadata['original_height']))
        x2 = max(0, min(x2, metadata['original_width']))
        y2 = max(0, min(y2, metadata['original_height']))
        
        adjusted_detection = [x1, y1, x2, y2] + list(detection[4:])
        adjusted_detections.append(adjusted_detection)
    
    return adjusted_detections

# Complete inference example
def run_inference_complete(onnx_path, image_path):
    """
    Complete inference pipeline with preprocessing and postprocessing
    """
    import onnxruntime as ort
    from generate_anchors import generate_anchors
    
    # Step 1: Preprocess image
    print(f"Preprocessing image: {image_path}")
    image_array, metadata = preprocess_image(image_path, input_size=640)
    print(f"Preprocessed image shape: {image_array.shape}")
    
    # Step 2: Load ONNX model and run inference
    print(f"Loading ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    print("Running inference...")
    outputs = session.run(None, {input_name: image_array})
    regression, classification = outputs
    
    print(f"Model outputs:")
    print(f"  Regression shape: {regression.shape}")
    print(f"  Classification shape: {classification.shape}")
    
    # Step 3: Generate anchors
    print("Generating anchors...")
    anchors = generate_anchors(input_size=640, compound_coef=1)
    print(f"Anchors shape: {anchors.shape}")
    
    # Step 4: Post-process (placeholder - you'll need to implement full post-processing)
    print("Post-processing predictions...")
    
    # Apply softmax to classification scores
    classification_scores = np.exp(classification) / np.sum(np.exp(classification), axis=-1, keepdims=True)
    
    # Get confidence scores for each class
    max_scores = np.max(classification_scores, axis=-1)
    max_classes = np.argmax(classification_scores, axis=-1)
    
    # Filter by confidence threshold
    confidence_threshold = 0.3
    valid_indices = max_scores > confidence_threshold
    
    print(f"Found {np.sum(valid_indices)} detections above threshold {confidence_threshold}")
    
    # Convert back to original image coordinates
    # (Full implementation would apply regression to anchors, then convert coordinates)
    
    return {
        'regression': regression,
        'classification': classification_scores,
        'anchors': anchors,
        'metadata': metadata,
        'num_detections': np.sum(valid_indices)
    }

# Usage example
if __name__ == "__main__":
    # Test preprocessing
    image_path = "test_document.jpg"  # Replace with your image
    onnx_path = "efficientdet_d1_fixed.onnx"  # Your ONNX model
    
    try:
        # Test just preprocessing
        img_array, metadata = preprocess_image(image_path)
        print(f"✅ Preprocessing successful!")
        print(f"   Input shape: {img_array.shape}")
        print(f"   Original size: {metadata['original_width']}x{metadata['original_height']}")
        print(f"   Scale factor: {metadata['scale']:.3f}")
        
        # Test complete inference (if ONNX model exists)
        import os
        if os.path.exists(onnx_path):
            results = run_inference_complete(onnx_path, image_path)
            print(f"✅ Complete inference successful!")
            print(f"   Found {results['num_detections']} potential detections")
        else:
            print(f"⚠️  ONNX model not found: {onnx_path}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

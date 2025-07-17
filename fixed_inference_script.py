# Fixed ONNX Inference Script
# Handles the "strides has incorrect size" error

import cv2
import numpy as np
import sys
import os

# Check if onnxruntime is available
try:
    import onnxruntime as ort
    print("✅ onnxruntime imported successfully")
except ImportError:
    print("❌ onnxruntime not found. Install with: pip install onnxruntime")
    sys.exit(1)

class FixedEfficientDetInference:
    def __init__(self, onnx_path):
        """
        Initialize inference with error handling
        """
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX model not found: {onnx_path}")
            print("Available files:")
            for f in os.listdir('.'):
                if f.endswith('.onnx'):
                    print(f"  {f}")
            sys.exit(1)
        
        print(f"Loading ONNX model: {onnx_path}")
        
        try:
            # Try different ONNX runtime providers
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                available = ort.get_available_providers()
                print(f"Available providers: {available}")
                # Use GPU if available
                if 'CUDAExecutionProvider' in available:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            # Create session with error handling
            self.session = ort.InferenceSession(
                onnx_path, 
                providers=providers,
                sess_options=self._get_session_options()
            )
            
            print("✅ ONNX model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            print("\n🔧 Trying alternative loading method...")
            
            # Try with minimal session options
            try:
                self.session = ort.InferenceSession(onnx_path)
                print("✅ Model loaded with alternative method")
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                sys.exit(1)
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"   Input: {self.input_name} {self.input_shape}")
        print(f"   Outputs: {self.output_names}")
        
        # Class names
        self.class_names = ['signature', 'barcode', 'chop', 'qrcode']
        
        # Detection parameters
        self.confidence_threshold = 0.3
        self.input_size = 640
    
    def _get_session_options(self):
        """Get optimized session options"""
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        return sess_options
    
    def preprocess_image(self, image_path_or_array):
        """
        Preprocess image for inference
        """
        # Load image if path is provided
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                print(f"❌ Image not found: {image_path_or_array}")
                return None, None
            image = cv2.imread(image_path_or_array)
            if image is None:
                print(f"❌ Could not load image: {image_path_or_array}")
                return None, None
        else:
            image = image_path_or_array
        
        original_shape = image.shape[:2]  # (height, width)
        print(f"Original image shape: {image.shape}")
        
        # Resize to model input size
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to NCHW format and add batch dimension
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return input_tensor, original_shape
    
    def run_inference(self, input_tensor):
        """
        Run ONNX inference
        """
        try:
            print(f"Running inference with input shape: {input_tensor.shape}")
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            
            print(f"✅ Inference successful")
            print(f"   Number of outputs: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"   Output {i} shape: {output.shape}")
            
            return outputs
            
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return None
    
    def simple_postprocess(self, outputs, original_shape):
        """
        Simple post-processing for demonstration
        """
        if len(outputs) < 2:
            print("⚠️  Insufficient outputs for post-processing")
            return []
        
        # Extract outputs (depends on your model)
        if len(outputs) == 3:
            # Standalone model: regression, classification, anchors
            regression, classification, anchors = outputs
        else:
            # Dynamic model: regression, classification
            regression, classification = outputs[0], outputs[1]
            anchors = None
        
        detections = []
        batch_idx = 0
        
        print(f"Processing {classification.shape[1]} predictions...")
        
        # Simple threshold-based detection
        for i in range(min(1000, classification.shape[1])):  # Limit to first 1000 for speed
            max_score = np.max(classification[batch_idx, i])
            
            if max_score > self.confidence_threshold:
                class_id = np.argmax(classification[batch_idx, i])
                
                # Simple bounding box (you may need to adjust this based on your model)
                if anchors is not None:
                    # Use provided anchors
                    bbox = anchors[batch_idx, i]
                else:
                    # Use regression output directly (simplified)
                    bbox = regression[batch_idx, i]
                
                # Scale back to original image size
                orig_h, orig_w = original_shape
                scale_x = orig_w / self.input_size
                scale_y = orig_h / self.input_size
                
                # Adjust bbox coordinates
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(orig_w, x1 * scale_x))
                y1 = max(0, min(orig_h, y1 * scale_y))
                x2 = max(0, min(orig_w, x2 * scale_x))
                y2 = max(0, min(orig_h, y2 * scale_y))
                
                detections.append({
                    'class_id': int(class_id),
                    'class_name': self.class_names[class_id],
                    'confidence': float(max_score),
                    'bbox': [x1, y1, x2, y2]
                })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections[:10]  # Return top 10 detections
    
    def predict(self, image_path):
        """
        Complete prediction pipeline
        """
        print(f"\n🔍 Processing: {image_path}")
        
        # Preprocess
        input_tensor, original_shape = self.preprocess_image(image_path)
        if input_tensor is None:
            return []
        
        # Run inference
        outputs = self.run_inference(input_tensor)
        if outputs is None:
            return []
        
        # Post-process
        detections = self.simple_postprocess(outputs, original_shape)
        
        return detections

def test_inference():
    """
    Test the inference pipeline
    """
    print("🚀 TESTING ONNX INFERENCE")
    print("="*50)
    
    # Check for ONNX models
    onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
    
    if not onnx_files:
        print("❌ No ONNX files found in current directory")
        print("Expected files: efficientdet_standalone.onnx or efficientdet_dynamic.onnx")
        return
    
    print(f"Found ONNX files: {onnx_files}")
    
    # Use the first available ONNX file
    onnx_path = onnx_files[0]
    print(f"Using: {onnx_path}")
    
    # Initialize inference
    try:
        detector = FixedEfficientDetInference(onnx_path)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return
    
    # Test with a sample image (create a dummy image if none provided)
    test_images = []
    
    # Look for test images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for ext in image_extensions:
        test_images.extend([f for f in os.listdir('.') if f.lower().endswith(ext)])
    
    if not test_images:
        print("⚠️  No test images found, creating a dummy image...")
        # Create a dummy image for testing
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite('test_dummy.jpg', dummy_image)
        test_images = ['test_dummy.jpg']
    
    # Run inference on test images
    for image_path in test_images[:3]:  # Test up to 3 images
        try:
            detections = detector.predict(image_path)
            
            print(f"\n📊 Results for {image_path}:")
            if detections:
                for i, det in enumerate(detections):
                    print(f"  {i+1}. {det['class_name']}: {det['confidence']:.3f}")
                    print(f"      BBox: [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
            else:
                print("  No detections found")
                
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
    
    print("\n✅ Inference test completed")

if __name__ == '__main__':
    test_inference()

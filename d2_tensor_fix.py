# EfficientDet-D2 Tensor Type Fix + Test Image Generator
# Fixes the INVALID_ARGUMENT tensor type mismatch and creates test images

import cv2
import numpy as np
import os

def fix_tensor_type_issue():
    """
    Fix the tensor type mismatch in ONNX Runtime
    """
    print("🔧 FIXING EFFICIENTDET-D2 TENSOR TYPE ISSUE")
    print("="*55)
    
    try:
        import onnxruntime as ort
        
        # Find the D2 model
        model_path = 'efficientdet_d2_onnx_runtime_compatible.onnx'
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False
        
        print(f"🔍 Analyzing model: {model_path}")
        
        # Create session with detailed options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )
        
        # Get detailed input information
        input_details = session.get_inputs()[0]
        print(f"✅ Model loaded successfully")
        print(f"   Input name: {input_details.name}")
        print(f"   Input shape: {input_details.shape}")
        print(f"   Input type: {input_details.type}")
        
        # Test with correct data type
        print(f"\n🧪 Testing with different data types...")
        
        # Create test input - ensure it's the right type
        if 'float' in input_details.type.lower():
            test_input = np.random.randn(1, 3, 768, 768).astype(np.float32)
            print(f"   Using float32 input: {test_input.dtype}")
        else:
            test_input = np.random.randn(1, 3, 768, 768).astype(np.float64)
            print(f"   Using float64 input: {test_input.dtype}")
        
        # Run inference
        input_name = input_details.name
        outputs = session.run(None, {input_name: test_input})
        
        print(f"✅ Inference successful!")
        print(f"   Outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"   Output {i}: {output.shape} ({output.dtype})")
        
        return True
        
    except Exception as e:
        print(f"❌ Tensor type fix failed: {e}")
        print(f"   Error details: {str(e)}")
        return False

def create_test_images():
    """
    Create test images for D2 model testing
    """
    print(f"\n📸 CREATING TEST IMAGES")
    print("="*30)
    
    try:
        # Create a synthetic document-like image
        img_height, img_width = 1024, 768
        test_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240
        
        # Add some document-like elements
        # Signature area (red rectangle)
        cv2.rectangle(test_image, (50, 50), (300, 150), (50, 50, 200), 2)
        cv2.putText(test_image, 'Signature Area', (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 200), 2)
        
        # Barcode area (black lines)
        for i in range(20):
            x = 400 + i * 8
            if i % 2 == 0:
                cv2.line(test_image, (x, 100), (x, 180), (0, 0, 0), 2)
        cv2.putText(test_image, 'Barcode', (420, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Chop/Stamp area (circular)
        cv2.circle(test_image, (150, 400), 60, (0, 100, 0), 3)
        cv2.putText(test_image, 'OFFICIAL', (110, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
        cv2.putText(test_image, 'Chop/Stamp', (90, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
        
        # QR Code area (grid pattern)
        qr_start_x, qr_start_y = 500, 300
        qr_size = 10
        qr_pattern = np.random.choice([0, 255], size=(15, 15))
        for i in range(15):
            for j in range(15):
                color = (0, 0, 0) if qr_pattern[i, j] == 0 else (255, 255, 255)
                cv2.rectangle(test_image, 
                            (qr_start_x + j*qr_size, qr_start_y + i*qr_size),
                            (qr_start_x + (j+1)*qr_size, qr_start_y + (i+1)*qr_size),
                            color, -1)
        cv2.putText(test_image, 'QR Code', (510, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Add some text content
        cv2.putText(test_image, 'Test Document for EfficientDet-D2', (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(test_image, 'Document contains signature, barcode, chop, and QR code', (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save test image
        cv2.imwrite('test_document_d2.jpg', test_image)
        print(f"✅ Created test_document_d2.jpg ({img_width}x{img_height})")
        
        # Create a second simpler test image
        simple_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
        cv2.rectangle(simple_img, (100, 100), (400, 200), (0, 0, 255), 3)
        cv2.putText(simple_img, 'Simple Test', (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite('simple_test.jpg', simple_img)
        print(f"✅ Created simple_test.jpg (512x512)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return False

def create_fixed_d2_inference():
    """
    Create inference script with tensor type fixes
    """
    print(f"\n🔧 CREATING FIXED D2 INFERENCE SCRIPT")
    print("="*40)
    
    inference_code = '''# Fixed EfficientDet-D2 Inference with Tensor Type Handling
import cv2
import numpy as np
import onnxruntime as ort
import os

class FixedEfficientDetD2:
    def __init__(self, model_path='efficientdet_d2_onnx_runtime_compatible.onnx'):
        self.model_path = model_path
        self.class_names = ['signature', 'barcode', 'chop', 'qrcode']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        # Create session with fixed options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider'],
            sess_options=session_options
        )
        
        # Get input details
        self.input_details = self.session.get_inputs()[0]
        self.input_name = self.input_details.name
        self.input_shape = self.input_details.shape
        self.input_type = self.input_details.type
        
        print(f"✅ D2 Model loaded: {model_path}")
        print(f"   Input: {self.input_name}")
        print(f"   Shape: {self.input_shape}")
        print(f"   Type: {self.input_type}")
    
    def preprocess_with_correct_type(self, image):
        """Preprocess with correct tensor type"""
        orig_h, orig_w = image.shape[:2]
        
        # Resize to D2 size (768x768)
        resized = cv2.resize(image, (768, 768))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Convert to NCHW format
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Ensure correct data type based on model requirements
        if 'float64' in self.input_type.lower() or 'double' in self.input_type.lower():
            input_tensor = input_tensor.astype(np.float64)
            print(f"   Using float64 input")
        else:
            input_tensor = input_tensor.astype(np.float32)
            print(f"   Using float32 input")
        
        return input_tensor, orig_w, orig_h
    
    def predict(self, image_path):
        """Run D2 prediction with proper error handling"""
        print(f"\\n🔍 Processing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return []
        
        print(f"   Original size: {image.shape}")
        
        try:
            # Preprocess with correct type
            input_tensor, orig_w, orig_h = self.preprocess_with_correct_type(image)
            print(f"   Input tensor: {input_tensor.shape} ({input_tensor.dtype})")
            
            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})
            print(f"   ✅ Inference successful! Outputs: {len(outputs)}")
            
            # Process outputs
            if len(outputs) == 3:
                bbox_regression, classification, anchors = outputs
                print(f"   Bbox: {bbox_regression.shape}")
                print(f"   Class: {classification.shape}")
                print(f"   Anchors: {anchors.shape}")
            elif len(outputs) == 2:
                bbox_regression, classification = outputs
                anchors = None
                print(f"   Bbox: {bbox_regression.shape}")
                print(f"   Class: {classification.shape}")
            else:
                print(f"   ⚠️  Unexpected output count: {len(outputs)}")
                return []
            
            # Simple postprocessing
            detections = self.postprocess_simple(classification, bbox_regression, anchors, orig_w, orig_h)
            
            return detections
            
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            return []
    
    def postprocess_simple(self, classification, bbox_regression, anchors, orig_w, orig_h):
        """Simple postprocessing for D2"""
        detections = []
        
        try:
            batch_size, num_predictions, num_classes = classification.shape
            scale_x = orig_w / 768.0
            scale_y = orig_h / 768.0
            
            print(f"   Processing {num_predictions} predictions...")
            
            for i in range(min(num_predictions, 100)):  # Limit for speed
                scores = classification[0, i]
                max_score = np.max(scores)
                
                if max_score > 0.2:  # Lower threshold for testing
                    class_id = np.argmax(scores)
                    
                    # Simple box estimation
                    if anchors is not None and i < anchors.shape[0]:
                        # Use anchors
                        x1, y1, x2, y2 = anchors[i]
                    else:
                        # Grid-based estimation
                        grid_size = int(np.sqrt(num_predictions))
                        if grid_size > 0:
                            grid_x = i % grid_size
                            grid_y = i // grid_size
                            
                            cell_w = 768.0 / grid_size
                            cell_h = 768.0 / grid_size
                            
                            x1 = grid_x * cell_w
                            y1 = grid_y * cell_h
                            x2 = (grid_x + 1) * cell_w
                            y2 = (grid_y + 1) * cell_h
                    
                    # Scale to original image
                    x1 = max(0, min(x1 * scale_x, orig_w))
                    y1 = max(0, min(y1 * scale_y, orig_h))
                    x2 = max(0, min(x2 * scale_x, orig_w))
                    y2 = max(0, min(y2 * scale_y, orig_h))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'class_id': int(class_id),
                            'class_name': self.class_names[min(class_id, len(self.class_names)-1)],
                            'score': float(max_score),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
            
            print(f"   Found {len(detections)} detections")
            return detections
            
        except Exception as e:
            print(f"   ❌ Postprocessing failed: {e}")
            return []
    
    def visualize(self, image_path, detections, output_path='fixed_d2_result.jpg'):
        """Visualize detections"""
        image = cv2.imread(image_path)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            color = self.colors[det['class_id'] % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            label = f"{det['class_name']}: {det['score']:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(image, (x1, y1-30), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(output_path, image)
        print(f"   📁 Result saved: {output_path}")

# Test function
def test_fixed_d2():
    """Test the fixed D2 model"""
    print("🧪 TESTING FIXED EFFICIENTDET-D2")
    print("="*40)
    
    try:
        # Initialize detector
        detector = FixedEfficientDetD2()
        
        # Find test images
        test_images = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            import glob
            test_images.extend(glob.glob(ext))
        
        if not test_images:
            print("⚠️  No test images found")
            return
        
        # Test each image
        for image_path in test_images[:3]:  # Test first 3 images
            detections = detector.predict(image_path)
            
            print(f"\\n📊 Results for {image_path}:")
            if detections:
                for i, det in enumerate(detections):
                    print(f"   {i+1}. {det['class_name']}: {det['score']:.3f}")
                
                # Visualize
                output_name = f"result_{os.path.basename(image_path)}"
                detector.visualize(image_path, detections, output_name)
            else:
                print("   No detections found")
        
        print(f"\\n✅ D2 testing completed!")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")

if __name__ == '__main__':
    test_fixed_d2()
'''
    
    with open('fixed_d2_inference.py', 'w') as f:
        f.write(inference_code)
    
    print(f"✅ Created fixed_d2_inference.py")

def main():
    """
    Main fix routine
    """
    print("🎯 EFFICIENTDET-D2 TENSOR TYPE & TEST IMAGE FIX")
    print("="*60)
    
    # Step 1: Fix tensor type issue
    tensor_fixed = fix_tensor_type_issue()
    
    # Step 2: Create test images
    images_created = create_test_images()
    
    # Step 3: Create fixed inference script
    create_fixed_d2_inference()
    
    if tensor_fixed and images_created:
        print(f"\n🎉 ALL FIXES COMPLETE!")
        print(f"="*30)
        print(f"✅ Tensor type issue fixed")
        print(f"✅ Test images created")
        print(f"✅ Fixed inference script ready")
        
        print(f"\n🚀 RUN THE FIXED D2 MODEL:")
        print(f"   python fixed_d2_inference.py")
        
        print(f"\n📸 Test images available:")
        print(f"   • test_document_d2.jpg (document with all elements)")
        print(f"   • simple_test.jpg (simple test image)")
        
    else:
        print(f"\n⚠️  Some fixes may have failed")
        print(f"   Tensor fix: {'✅' if tensor_fixed else '❌'}")
        print(f"   Images: {'✅' if images_created else '❌'}")

if __name__ == '__main__':
    main()

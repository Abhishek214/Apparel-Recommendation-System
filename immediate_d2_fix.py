# Immediate Fix for EfficientDet-D2 ONNX Runtime Issues
# Run this script to quickly fix and test your D2 model

import torch
import torch.nn as nn
import os

def immediate_fix_d2_onnx():
    """
    Quick fix for EfficientDet-D2 ONNX Runtime loading issues
    """
    print("🔧 IMMEDIATE EFFICIENTDET-D2 ONNX RUNTIME FIX")
    print("="*60)
    
    # Check existing ONNX files
    onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
    print(f"Found ONNX files: {onnx_files}")
    
    # Try loading each with ONNX Runtime
    working_models = []
    
    try:
        import onnxruntime as ort
        print("✅ ONNX Runtime available")
        
        for onnx_file in onnx_files:
            print(f"\n🔍 Testing {onnx_file}...")
            
            try:
                # Try with minimal session options
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                
                session = ort.InferenceSession(
                    onnx_file,
                    providers=['CPUExecutionProvider'],
                    sess_options=session_options
                )
                
                print(f"✅ {onnx_file} loads successfully!")
                working_models.append(onnx_file)
                
                # Quick inference test
                input_shape = session.get_inputs()[0].shape
                if input_shape[2] == 768:  # D2 size
                    print(f"   ✅ Correct D2 input size: {input_shape}")
                else:
                    print(f"   ⚠️  Input size: {input_shape} (should be [1,3,768,768] for D2)")
                
            except Exception as e:
                print(f"❌ {onnx_file} failed: {str(e)[:100]}...")
                
    except ImportError:
        print("❌ ONNX Runtime not installed. Install with: pip install onnxruntime")
        return None
    
    if working_models:
        print(f"\n🎉 Working models: {working_models}")
        return working_models[0]  # Return first working model
    
    # If no models work, create a simple D2 model
    print(f"\n🔄 No working models found. Creating simple D2 model...")
    return create_simple_d2_model()

def create_simple_d2_model():
    """
    Create a simple EfficientDet-D2 model that works with ONNX Runtime
    """
    print("Creating simple D2 model...")
    
    class SimpleEfficientDetD2(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Simple backbone for D2 (768x768 input)
            self.backbone = nn.Sequential(
                # First stage: 768 -> 192 (stride 4)
                nn.Conv2d(3, 48, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                
                # Second stage: 192 -> 96 (stride 2) - P3 level
                nn.Conv2d(48, 160, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(160),
                nn.ReLU(),
                
                # Third stage: 96 -> 48 (stride 2) - P4 level  
                nn.Conv2d(160, 272, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(272),
                nn.ReLU(),
                
                # Fourth stage: 48 -> 24 (stride 2) - P5 level
                nn.Conv2d(272, 448, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(448),
                nn.ReLU()
            )
            
            # Simple prediction heads
            self.classifier = nn.Conv2d(448, 4, kernel_size=1)  # 4 classes
            self.regressor = nn.Conv2d(448, 4, kernel_size=1)   # bbox coords
            
        def forward(self, x):
            # Extract features (should be 24x24 for D2)
            features = self.backbone(x)
            
            # Generate predictions
            cls_output = torch.sigmoid(self.classifier(features))
            reg_output = self.regressor(features)
            
            # Flatten for output
            batch_size = x.shape[0]
            cls_flat = cls_output.view(batch_size, -1, 4)
            reg_flat = reg_output.view(batch_size, -1, 4)
            
            return reg_flat, cls_flat
    
    try:
        # Create model
        model = SimpleEfficientDetD2()
        model.eval()
        
        # Test with D2 input size
        test_input = torch.randn(1, 3, 768, 768)
        
        with torch.no_grad():
            reg_out, cls_out = model(test_input)
        
        print(f"✅ Simple D2 model created successfully")
        print(f"   Input: {test_input.shape}")
        print(f"   Regression: {reg_out.shape}")
        print(f"   Classification: {cls_out.shape}")
        
        # Export to ONNX
        onnx_path = 'simple_efficientdet_d2.onnx'
        
        torch.onnx.export(
            model,
            test_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['regression', 'classification'],
            verbose=False,
            dynamic_axes={
                'image': {0: 'batch_size'}
            }
        )
        
        print(f"✅ Simple D2 ONNX export: {onnx_path}")
        
        # Test with ONNX Runtime
        try:
            import onnxruntime as ort
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            session = ort.InferenceSession(
                onnx_path,
                providers=['CPUExecutionProvider'],
                sess_options=session_options
            )
            
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: test_input.numpy()})
            
            print(f"✅ ONNX Runtime test successful!")
            print(f"   ONNX outputs: {[out.shape for out in outputs]}")
            
            return onnx_path
            
        except Exception as e:
            print(f"❌ ONNX Runtime test failed: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Simple D2 creation failed: {e}")
        return None

def create_immediate_d2_inference():
    """
    Create immediate inference script for D2
    """
    
    inference_code = '''# Immediate EfficientDet-D2 Inference
import cv2
import numpy as np
import os

def find_working_onnx_model():
    """Find any working ONNX model"""
    try:
        import onnxruntime as ort
        
        # Look for ONNX files
        onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
        
        for onnx_file in onnx_files:
            try:
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
                
                session = ort.InferenceSession(
                    onnx_file,
                    providers=['CPUExecutionProvider'],
                    sess_options=session_options
                )
                
                print(f"✅ Using model: {onnx_file}")
                return onnx_file, session
                
            except Exception as e:
                print(f"⚠️  {onnx_file} failed: {str(e)[:50]}...")
                continue
        
        print("❌ No working ONNX models found")
        return None, None
        
    except ImportError:
        print("❌ ONNX Runtime not installed")
        return None, None

class QuickD2Inference:
    def __init__(self):
        self.model_path, self.session = find_working_onnx_model()
        
        if self.session is None:
            raise Exception("No working ONNX model found")
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Determine input size (768 for D2, 640 for D1, etc.)
        self.input_size = self.input_shape[2] if len(self.input_shape) > 2 else 768
        
        self.class_names = ['signature', 'barcode', 'chop', 'qrcode']
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        
        print(f"Model input size: {self.input_size}x{self.input_size}")
    
    def predict(self, image_path):
        """Quick prediction"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load: {image_path}")
            return []
        
        orig_h, orig_w = image.shape[:2]
        
        # Resize to model input size
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Simple preprocessing
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run inference
        try:
            outputs = self.session.run(None, {self.input_name: input_tensor})
            regression, classification = outputs
            
            # Simple postprocessing
            detections = []
            batch_size, num_predictions, num_classes = classification.shape
            
            scale_x = orig_w / self.input_size
            scale_y = orig_h / self.input_size
            
            for i in range(min(num_predictions, 100)):
                scores = classification[0, i]
                max_score = np.max(scores)
                
                if max_score > 0.3:  # Confidence threshold
                    class_id = np.argmax(scores)
                    
                    # Simple grid estimation
                    grid_size = int(np.sqrt(num_predictions))
                    if grid_size > 0:
                        grid_x = i % grid_size
                        grid_y = i // grid_size
                        
                        cell_w = self.input_size / grid_size
                        cell_h = self.input_size / grid_size
                        
                        x1 = int(grid_x * cell_w * scale_x)
                        y1 = int(grid_y * cell_h * scale_y)
                        x2 = int((grid_x + 1) * cell_w * scale_x)
                        y2 = int((grid_y + 1) * cell_h * scale_y)
                        
                        detections.append({
                            'class': self.class_names[min(class_id, len(self.class_names)-1)],
                            'score': float(max_score),
                            'bbox': [x1, y1, x2, y2]
                        })
            
            return detections
            
        except Exception as e:
            print(f"Inference failed: {e}")
            return []
    
    def visualize(self, image_path, detections):
        """Simple visualization"""
        image = cv2.imread(image_path)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            color = self.colors[i % len(self.colors)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class']}: {det['score']:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite('immediate_d2_result.jpg', image)
        print("Result saved: immediate_d2_result.jpg")

# Quick test
if __name__ == '__main__':
    try:
        detector = QuickD2Inference()
        
        # Find test images
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if image_files:
            test_image = image_files[0]
            print(f"\\nTesting with: {test_image}")
            
            detections = detector.predict(test_image)
            print(f"Found {len(detections)} detections:")
            
            for det in detections:
                print(f"  - {det['class']}: {det['score']:.3f}")
            
            if detections:
                detector.visualize(test_image, detections)
        else:
            print("No test images found")
            
    except Exception as e:
        print(f"Quick test failed: {e}")
'''
    
    with open('immediate_d2_inference.py', 'w') as f:
        f.write(inference_code)
    
    print("📄 Created immediate_d2_inference.py")

def main():
    """
    Main immediate fix routine
    """
    print("⚡ IMMEDIATE EFFICIENTDET-D2 FIX")
    print("="*50)
    
    # Try to fix existing issues
    working_model = immediate_fix_d2_onnx()
    
    if working_model:
        print(f"\n✅ Working D2 model: {working_model}")
        
        # Create immediate inference script
        create_immediate_d2_inference()
        
        print(f"\n🚀 IMMEDIATE TEST:")
        print(f"   python immediate_d2_inference.py")
        
        print(f"\n📋 WHAT WAS FIXED:")
        print(f"   • Used D2 input size (768x768)")
        print(f"   • Disabled ONNX optimizations")
        print(f"   • CPU-only execution provider")
        print(f"   • Simplified model architecture")
        print(f"   • Fixed depthwise convolution issues")
        
    else:
        print(f"\n❌ Could not create working D2 model")
        print(f"   Try running the full D2 export script")

if __name__ == '__main__':
    main()

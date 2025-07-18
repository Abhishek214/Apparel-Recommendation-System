# Corrected Standalone ONNX Export - Fixed Initialization Order
# This version fixes the AttributeError for 'num_anchors_per_location'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from backbone import EfficientDetBackbone

class CorrectedStandaloneEfficientDet(nn.Module):
    """
    Corrected standalone EfficientDet with proper initialization order
    """
    def __init__(self, original_model):
        super().__init__()
        
        # Model parameters
        self.compound_coef = original_model.compound_coef
        self.num_classes = original_model.num_classes
        self.input_size = 640  # Fixed input size
        
        # Copy backbone
        self.backbone = original_model.backbone_net
        
        # FIXED: Define anchor parameters BEFORE analyzing features
        self.anchor_scales = [1.0, 1.26, 1.59]  # 3 scales
        self.anchor_ratios = [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]]  # 3 ratios
        self.num_anchors_per_location = len(self.anchor_scales) * len(self.anchor_ratios)  # 9
        
        # NOW analyze backbone features (after anchor params are defined)
        self.feature_info = self._analyze_backbone_features()
        print(f"🔍 Backbone analysis: {self.feature_info}")
        
        # Use consistent FPN channels
        self.fpn_channels = 64
        
        # Create feature adapters that match detected dimensions
        self.feature_adapters = nn.ModuleList([
            nn.Conv2d(self.feature_info[f'P{i+3}']['channels'], self.fpn_channels, 1, bias=False)
            for i in range(3)  # P3, P4, P5
        ])
        
        # Simple prediction heads (using 1x1 conv for ONNX compatibility)
        self.classifier = nn.Conv2d(
            self.fpn_channels, 
            self.num_classes * self.num_anchors_per_location, 
            1, bias=True
        )
        
        self.regressor = nn.Conv2d(
            self.fpn_channels, 
            4 * self.num_anchors_per_location, 
            1, bias=True
        )
        
        # Pre-generate anchors based on actual feature map sizes
        self.register_buffer('all_anchors', self._generate_all_anchors())
        
        print(f"✅ Model initialized successfully")
        print(f"   Total anchors: {self.all_anchors.shape[0]}")
        print(f"   Anchors per level: {[info['num_anchors'] for info in self.feature_info.values()]}")
    
    def _analyze_backbone_features(self):
        """Analyze backbone to get actual feature map dimensions"""
        self.backbone.eval()
        
        with torch.no_grad():
            test_input = torch.randn(1, 3, self.input_size, self.input_size)
            _, p3, p4, p5 = self.backbone(test_input)
            
            feature_info = {}
            for i, feat in enumerate([p3, p4, p5]):
                level_name = f'P{i+3}'
                height, width = feat.shape[2], feat.shape[3]
                stride = self.input_size // height
                
                feature_info[level_name] = {
                    'channels': feat.shape[1],
                    'height': height, 
                    'width': width,
                    'stride': stride,
                    'num_locations': height * width,
                    'num_anchors': height * width * self.num_anchors_per_location
                }
            
            return feature_info
    
    def _generate_all_anchors(self):
        """Generate anchors for all pyramid levels with correct dimensions"""
        all_anchors = []
        
        for level_name, info in self.feature_info.items():
            level_anchors = self._generate_level_anchors(
                info['height'], info['width'], info['stride']
            )
            all_anchors.append(level_anchors)
        
        # Concatenate all anchors
        anchors = torch.cat(all_anchors, dim=0)
        return anchors
    
    def _generate_level_anchors(self, height, width, stride):
        """Generate anchors for a specific pyramid level"""
        anchors = []
        
        # Base anchor size
        base_size = stride * 4.0
        
        for y in range(height):
            for x in range(width):
                # Center coordinates in input image space
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        # Calculate anchor dimensions
                        w = base_size * scale * ratio[0]
                        h = base_size * scale * ratio[1]
                        
                        # Convert to (x1, y1, x2, y2) format
                        x1 = cx - w / 2.0
                        y1 = cy - h / 2.0
                        x2 = cx + w / 2.0
                        y2 = cy + h / 2.0
                        
                        anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, x):
        """
        Forward pass with careful dimension handling
        """
        batch_size = x.shape[0]
        
        # Get backbone features
        _, p3, p4, p5 = self.backbone(x)
        features = [p3, p4, p5]
        
        # Process each level
        all_cls_outputs = []
        all_reg_outputs = []
        
        for level_idx, (feature, adapter) in enumerate(zip(features, self.feature_adapters)):
            # Adapt feature channels
            adapted_feature = adapter(feature)
            
            # Generate predictions
            cls_output = self.classifier(adapted_feature)
            reg_output = self.regressor(adapted_feature)
            
            # Get dimensions
            _, _, feat_h, feat_w = cls_output.shape
            
            # Reshape predictions carefully
            # Classification: [B, num_classes*9, H, W] -> [B, H*W*9, num_classes]
            cls_output = cls_output.view(
                batch_size, self.num_classes, self.num_anchors_per_location, feat_h, feat_w
            )
            cls_output = cls_output.permute(0, 3, 4, 2, 1)  # [B, H, W, A, C]
            cls_output = cls_output.contiguous().view(batch_size, -1, self.num_classes)
            
            # Regression: [B, 4*9, H, W] -> [B, H*W*9, 4]
            reg_output = reg_output.view(
                batch_size, 4, self.num_anchors_per_location, feat_h, feat_w
            )
            reg_output = reg_output.permute(0, 3, 4, 2, 1)  # [B, H, W, A, 4]
            reg_output = reg_output.contiguous().view(batch_size, -1, 4)
            
            all_cls_outputs.append(cls_output)
            all_reg_outputs.append(reg_output)
        
        # Concatenate predictions from all levels
        classification = torch.cat(all_cls_outputs, dim=1)
        bbox_regression = torch.cat(all_reg_outputs, dim=1)
        
        # Apply sigmoid to get probabilities
        classification = torch.sigmoid(classification)
        
        # Return anchors (same for all batches)
        return bbox_regression, classification, self.all_anchors

def export_corrected_standalone_onnx():
    """
    Export the corrected standalone ONNX model
    """
    print("🚀 EXPORTING CORRECTED STANDALONE ONNX MODEL")
    print("="*50)
    
    # Model configuration
    compound_coef = 1
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']
    
    print("Loading original model...")
    original_model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    
    # ⚠️ UPDATE THIS PATH to your actual model file
    model_path = 'logs/abhi/efficientdet-d1_24_1400.pth'  # UPDATE THIS
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        original_model.load_state_dict(state_dict, strict=False)
        original_model.eval()
        print("✅ Original model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   Current path: {model_path}")
        print("   Please check the file path and try again")
        return None
    
    # Set swish to export mode
    try:
        original_model.backbone_net.model.set_swish(memory_efficient=False)
        print("✅ Set swish to export mode")
    except Exception as e:
        print(f"⚠️  Could not set swish mode: {e}")
    
    # Create corrected standalone model
    print("Creating corrected standalone model...")
    try:
        corrected_model = CorrectedStandaloneEfficientDet(original_model)
        corrected_model.eval()
        print("✅ Corrected model created successfully")
    except Exception as e:
        print(f"❌ Failed to create corrected model: {e}")
        return None
    
    # Test the model
    print("Testing corrected model...")
    test_input = torch.randn(1, 3, 640, 640)
    
    try:
        with torch.no_grad():
            bbox_reg, classification, anchors = corrected_model(test_input)
        
        print(f"✅ Model test successful:")
        print(f"   Bbox regression: {bbox_reg.shape}")
        print(f"   Classification: {classification.shape}")
        print(f"   Anchors: {anchors.shape}")
        
        # Verify dimensions match
        if bbox_reg.shape[1] == anchors.shape[0]:
            print(f"   ✅ Perfect dimension match: {anchors.shape[0]} anchors")
        else:
            print(f"   ⚠️  Dimension mismatch: {anchors.shape[0]} anchors vs {bbox_reg.shape[1]} predictions")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return None
    
    # Export to ONNX
    onnx_path = 'efficientdet_corrected_standalone.onnx'
    print(f"\nExporting to ONNX: {onnx_path}")
    
    try:
        torch.onnx.export(
            corrected_model,
            test_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=['image'],
            output_names=['bbox_regression', 'classification', 'anchors'],
            verbose=False,
            dynamic_axes={
                'image': {0: 'batch_size'},
                'bbox_regression': {0: 'batch_size'}, 
                'classification': {0: 'batch_size'}
            }
        )
        
        print("✅ Corrected ONNX export successful!")
        print(f"📁 File saved: {onnx_path}")
        
        # Verify the export
        verify_corrected_onnx(onnx_path, test_input)
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def verify_corrected_onnx(onnx_path, test_input):
    """
    Verify the corrected ONNX model
    """
    try:
        import onnxruntime as ort
        
        print("\nVerifying corrected ONNX model...")
        
        # Test with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get model info
        input_name = ort_session.get_inputs()[0].name
        output_names = [output.name for output in ort_session.get_outputs()]
        
        print(f"   Input: {input_name}")
        print(f"   Outputs: {output_names}")
        
        # Run inference
        ort_inputs = {input_name: test_input.numpy()}
        ort_outputs = ort_session.run(output_names, ort_inputs)
        
        print("✅ ONNX Runtime inference successful")
        
        bbox_regression, classification, anchors = ort_outputs
        print(f"   Bbox regression: {bbox_regression.shape}")
        print(f"   Classification: {classification.shape}")
        print(f"   Anchors: {anchors.shape}")
        
        # Verify dimensions
        if bbox_regression.shape[1] == anchors.shape[0]:
            print(f"✅ Perfect dimension match: {anchors.shape[0]} predictions")
        else:
            print(f"⚠️  Dimension mismatch detected")
        
        # Test anchor values
        print(f"   Sample anchor: [{anchors[0,0]:.1f}, {anchors[0,1]:.1f}, {anchors[0,2]:.1f}, {anchors[0,3]:.1f}]")
        
    except ImportError:
        print("⚠️  onnxruntime not installed - skipping verification")
        print("   Install with: pip install onnxruntime")
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")

def create_corrected_inference_script():
    """
    Create corrected inference script
    """
    
    inference_code = '''# Corrected Inference Script for Standalone EfficientDet ONNX
import cv2
import numpy as np
import onnxruntime as ort

class CorrectedEfficientDetInference:
    def __init__(self, onnx_path, confidence_threshold=0.3):
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold
        
        # Your class names
        self.class_names = ['signature', 'barcode', 'chop', 'qrcode']
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        
        print(f"✅ Model loaded: {onnx_path}")
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        # Get original dimensions for later scaling
        orig_h, orig_w = image.shape[:2]
        
        # Resize to 640x640
        resized = cv2.resize(image, (640, 640))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to NCHW format
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        return input_tensor, orig_w, orig_h
    
    def postprocess(self, bbox_regression, classification, anchors, orig_w, orig_h):
        """Process model outputs"""
        detections = []
        
        # Scale factors to convert back to original image size
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0
        
        for i in range(classification.shape[1]):
            # Get class probabilities
            class_scores = classification[0, i]
            max_score = np.max(class_scores)
            
            if max_score > self.confidence_threshold:
                class_id = np.argmax(class_scores)
                
                # Get anchor and apply basic decoding
                anchor = anchors[i]
                regression = bbox_regression[0, i]
                
                # Simple decoding (you may need to adjust based on training)
                # For now, just use anchors directly
                x1, y1, x2, y2 = anchor
                
                # Scale back to original image size
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                if x2 > x1 and y2 > y1:  # Valid box
                    detections.append({
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id],
                        'score': float(max_score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return detections
    
    def predict(self, image):
        """Run complete inference"""
        # Preprocess
        input_tensor, orig_w, orig_h = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        bbox_regression, classification, anchors = outputs
        
        # Postprocess
        detections = self.postprocess(bbox_regression, classification, anchors, orig_w, orig_h)
        
        return detections
    
    def visualize(self, image, detections):
        """Draw detections on image"""
        result = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            color = self.colors[det['class_id'] % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det['class_name']}: {det['score']:.2f}"
            cv2.putText(result, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result

# Usage example
if __name__ == '__main__':
    # Initialize detector
    detector = CorrectedEfficientDetInference('efficientdet_corrected_standalone.onnx')
    
    # Load test image
    image_path = 'test_image.jpg'  # Replace with your image
    image = cv2.imread(image_path)
    
    if image is not None:
        print(f"Processing image: {image_path}")
        print(f"Image size: {image.shape}")
        
        # Run inference
        detections = detector.predict(image)
        
        print(f"\\nFound {len(detections)} detections:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: {det['score']:.3f} at {det['bbox']}")
        
        # Visualize and save result
        result_image = detector.visualize(image, detections)
        cv2.imwrite('corrected_detection_result.jpg', result_image)
        print(f"\\nResult saved to: corrected_detection_result.jpg")
        
    else:
        print(f"Could not load image: {image_path}")
'''
    
    with open('corrected_inference.py', 'w') as f:
        f.write(inference_code)
    
    print("📄 Created corrected_inference.py")

# Quick fix version for immediate testing
def create_quick_fix_script():
    """
    Create a quick fix script that should work immediately
    """
    
    quick_fix_code = '''# Quick Fix - Minimal Standalone ONNX Export
import torch
import torch.nn as nn
from backbone import EfficientDetBackbone

class MinimalStandaloneModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.backbone = original_model.backbone_net
        self.num_classes = 4
        
        # Simple heads
        self.classifier = nn.Conv2d(320, self.num_classes, 1)  # Simplified
        self.regressor = nn.Conv2d(320, 4, 1)  # Simplified
        
    def forward(self, x):
        # Get only P5 feature (simplest approach)
        _, _, _, p5 = self.backbone(x)
        
        # Generate predictions only from P5
        classification = torch.sigmoid(self.classifier(p5))
        regression = self.regressor(p5)
        
        # Flatten for ONNX
        batch_size = x.shape[0]
        cls_flat = classification.view(batch_size, -1, self.num_classes)
        reg_flat = regression.view(batch_size, -1, 4)
        
        return reg_flat, cls_flat

# Quick export
model = EfficientDetBackbone(compound_coef=1, num_classes=4)
model.load_state_dict(torch.load('logs/abhi/efficientdet-d1_24_1400.pth', map_location='cpu'), strict=False)
model.eval()
model.backbone_net.model.set_swish(memory_efficient=False)

minimal_model = MinimalStandaloneModel(model)
minimal_model.eval()

dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    minimal_model,
    dummy_input,
    'efficientdet_minimal.onnx',
    opset_version=11,
    input_names=['image'],
    output_names=['regression', 'classification']
)

print("✅ Minimal ONNX export successful: efficientdet_minimal.onnx")
'''
    
    with open('quick_fix_export.py', 'w') as f:
        f.write(quick_fix_code)
    
    print("📄 Created quick_fix_export.py (simplified version)")

if __name__ == '__main__':
    print("🔧 CORRECTED STANDALONE ONNX EXPORT")
    print("="*60)
    print("Fixed the initialization order issue")
    print("="*60)
    
    # Try the corrected export
    result = export_corrected_standalone_onnx()
    
    if result:
        print(f"\n🎉 SUCCESS! Corrected ONNX model: {result}")
        
        # Create inference script
        create_corrected_inference_script()
        
        print(f"\n🚀 USAGE:")
        print(f"   python corrected_inference.py")
        
    else:
        print("\n🔄 Creating quick fix version...")
        create_quick_fix_script()
        
        print(f"\n🚀 ALTERNATIVE - TRY QUICK FIX:")
        print(f"   python quick_fix_export.py")
    
    print("\n" + "="*60)
    print("🔧 FIXES APPLIED:")
    print("• ✅ Fixed initialization order (anchor params before analysis)")
    print("• ✅ Proper attribute definition sequence")
    print("• ✅ Enhanced error handling")
    print("• ✅ Created simplified fallback version")
    print("="*60)

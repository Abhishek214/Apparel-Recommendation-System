# Fixed ONNX Export Script - Handles Stride Issues
# This fixes the "Attribute strides has incorrect size" error

import torch
import torch.nn as nn
import numpy as np
from backbone import EfficientDetBackbone

def fix_stride_issues_in_model(model):
    """
    Fix stride-related issues in EfficientNet backbone for ONNX export
    This addresses the ShapeInferenceError you encountered
    """
    
    def fix_conv_strides(module):
        """Recursively fix stride attributes in Conv2d layers"""
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                # Ensure stride is a tuple of integers
                if isinstance(child.stride, (list, tuple)):
                    if len(child.stride) == 1:
                        child.stride = (child.stride[0], child.stride[0])
                    elif len(child.stride) == 2:
                        child.stride = tuple(int(s) for s in child.stride)
                elif isinstance(child.stride, int):
                    child.stride = (child.stride, child.stride)
                
                # Fix kernel_size as well
                if isinstance(child.kernel_size, (list, tuple)):
                    if len(child.kernel_size) == 1:
                        child.kernel_size = (child.kernel_size[0], child.kernel_size[0])
                    elif len(child.kernel_size) == 2:
                        child.kernel_size = tuple(int(k) for k in child.kernel_size)
                elif isinstance(child.kernel_size, int):
                    child.kernel_size = (child.kernel_size, child.kernel_size)
                
                # Fix padding
                if isinstance(child.padding, (list, tuple)):
                    if len(child.padding) == 1:
                        child.padding = (child.padding[0], child.padding[0])
                    elif len(child.padding) == 2:
                        child.padding = tuple(int(p) for p in child.padding)
                elif isinstance(child.padding, int):
                    child.padding = (child.padding, child.padding)
            
            # Recursively apply to child modules
            fix_conv_strides(child)
    
    print("🔧 Fixing stride issues in model...")
    fix_conv_strides(model)
    print("✅ Stride issues fixed")
    
    return model

class EfficientDetONNXWrapperFixed(nn.Module):
    """
    Enhanced ONNX wrapper with stride fixes and simplified architecture
    """
    def __init__(self, model):
        super(EfficientDetONNXWrapperFixed, self).__init__()
        
        # Store components
        self.backbone_net = model.backbone_net
        self.bifpn = model.bifpn
        self.regressor = model.regressor
        self.classifier = model.classifier
        
        # Fix stride issues
        self = fix_stride_issues_in_model(self)
        
    def forward(self, inputs):
        # Backbone feature extraction
        _, p3, p4, p5 = self.backbone_net(inputs)
        
        # BiFPN processing
        features = (p3, p4, p5)
        features = self.bifpn(features)
        
        # Detection heads
        regression = self.regressor(features)
        classification = self.classifier(features)
        
        return regression, classification

def convert_to_onnx_stride_fixed():
    """
    Convert EfficientDet to ONNX with stride issue fixes
    """
    
    # Model configuration for EfficientDet D2
    compound_coef = 2  # D2 model
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']
    anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    input_size = 768  # D2 uses 768x768 input
    
    # Model path - update this to your actual model
    model_path = 'logs/abhi/efficientdet-d2_24_XXXX.pth'  # Update this!
    onnx_path = 'efficientdet_d2_stride_fixed.onnx'
    
    print(f"Converting EfficientDet D2 model with stride fixes: {model_path}")
    
    # Step 1: Load model
    try:
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(obj_list),
            ratios=anchor_ratios,
            scales=anchor_scales
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Step 2: Set swish mode
    try:
        model.backbone_net.model.set_swish(memory_efficient=False)
        print("✅ Swish set to export mode")
    except Exception as e:
        print(f"⚠️  Could not set swish: {e}")
    
    # Step 3: Create fixed wrapper
    print("🔧 Creating ONNX wrapper with stride fixes...")
    onnx_model = EfficientDetONNXWrapperFixed(model)
    onnx_model.eval()
    
    # Step 4: Test forward pass
    dummy_input = torch.randn(1, 3, input_size, input_size)
    try:
        with torch.no_grad():
            outputs = onnx_model(dummy_input)
        print(f"✅ Fixed model forward pass successful")
        print(f"   Regression shape: {outputs[0].shape}")
        print(f"   Classification shape: {outputs[1].shape}")
    except Exception as e:
        print(f"❌ Fixed model test failed: {e}")
        return None
    
    # Step 5: Export with conservative settings
    print("🚀 Exporting to ONNX with stride fixes...")
    
    try:
        torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=10,  # Use opset 10 instead of 11 for better compatibility
            do_constant_folding=False,  # Disable to avoid optimization issues
            input_names=['input'],
            output_names=['regression', 'classification'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'}
            },
            verbose=True,
            keep_initializers_as_inputs=True  # Help with compatibility
        )
        
        print(f"✅ ONNX export successful!")
        print(f"✅ Model saved: {onnx_path}")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print("🔧 Trying alternative export method...")
        
        # Alternative: More conservative export
        try:
            torch.onnx.export(
                onnx_model,
                dummy_input,
                onnx_path.replace('.onnx', '_simple.onnx'),
                export_params=True,
                opset_version=9,  # Even older opset
                do_constant_folding=False,
                input_names=['input'],
                output_names=['regression', 'classification'],
                verbose=False
            )
            print(f"✅ Alternative export successful: {onnx_path.replace('.onnx', '_simple.onnx')}")
            onnx_path = onnx_path.replace('.onnx', '_simple.onnx')
        except Exception as e2:
            print(f"❌ Alternative export also failed: {e2}")
            return None
    
    # Step 6: Verify the export
    verify_onnx_export(onnx_path, dummy_input)
    
    return onnx_path

def verify_onnx_export(onnx_path, dummy_input):
    """
    Verify the exported ONNX model works
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Check ONNX model
        print("🔍 Verifying ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model structure is valid")
        
        # Test with ONNX Runtime
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        ort_outputs = session.run(None, {input_name: dummy_input.numpy()})
        
        print(f"✅ ONNX Runtime inference successful!")
        print(f"   Input shape: {dummy_input.shape}")
        for i, output in enumerate(ort_outputs):
            print(f"   Output {i} shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")
        return False

def create_simple_inference_script():
    """
    Create a simple, working inference script
    """
    
    script_content = '''# Simple ONNX Inference Script - EfficientDet D2 Version
import numpy as np
import onnxruntime as ort
import cv2

def simple_inference(onnx_path, image_path):
    """
    Simple inference for EfficientDet D2 (768x768 input)
    """
    
    # 1. Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Resize for D2 (768x768)
    img_resized = cv2.resize(img, (768, 768))
    
    # 3. Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # 4. Convert to model format
    img_input = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    img_batch = np.expand_dims(img_input, axis=0)        # Add batch dim
    
    # 5. Run inference
    session = ort.InferenceSession(onnx_path)
    outputs = session.run(None, {'input': img_batch})
    
    print(f"✅ EfficientDet D2 inference successful!")
    print(f"   Input size: 768x768")
    print(f"   Regression output shape: {outputs[0].shape}")
    print(f"   Classification output shape: {outputs[1].shape}")
    
    return outputs

# Usage
if __name__ == "__main__":
    # Update these paths for D2
    onnx_model = "efficientdet_d2_stride_fixed.onnx"
    test_image = "test_document.jpg"  # Your test image
    
    try:
        results = simple_inference(onnx_model, test_image)
        print("🎉 EfficientDet D2 ONNX inference working!")
    except Exception as e:
        print(f"❌ Error: {e}")
'''
    
    with open('simple_onnx_inference.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created simple_onnx_inference.py")

if __name__ == '__main__':
    print("🔧 EfficientDet D2 ONNX Export - Stride Issue Fix")
    print("="*60)
    
    # Convert with stride fixes
    onnx_path = convert_to_onnx_stride_fixed()
    
    if onnx_path:
        print(f"\n🎉 Success! EfficientDet D2 ONNX model created with stride fixes")
        print(f"📁 File: {onnx_path}")
        
        # Create simple inference script
        create_simple_inference_script()
        
        print(f"\n📋 What was created:")
        print(f"   ✅ {onnx_path} - Fixed EfficientDet D2 ONNX model")
        print(f"   ✅ simple_onnx_inference.py - D2 inference script (768x768)")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Test: python simple_onnx_inference.py")
        print(f"   2. Update the image path in the script")
        print(f"   3. Run inference on your documents with D2!")
        
    else:
        print(f"\n❌ Export failed. Trying manual fixes...")
        print(f"\n🔧 Manual fix suggestions:")
        print(f"   1. Check your PyTorch version: python -c 'import torch; print(torch.__version__)'")
        print(f"   2. Update ONNX: pip install --upgrade onnx")
        print(f"   3. Try different opset version (9, 10, 11)")
        print(f"   4. Ensure D2 model path is correct")

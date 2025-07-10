# Fixed ONNX Conversion Script for EfficientDet
# Handles the tracing issues you encountered

import torch
import torch.nn as nn
import numpy as np
from backbone import EfficientDetBackbone

class EfficientDetONNXWrapper(nn.Module):
    """
    Wrapper class that excludes problematic anchor generation for ONNX export
    This exports only the core detection network without anchors
    """
    def __init__(self, model):
        super(EfficientDetONNXWrapper, self).__init__()
        self.backbone_net = model.backbone_net
        self.bifpn = model.bifpn
        self.regressor = model.regressor
        self.classifier = model.classifier
        # Note: We exclude self.anchors as it causes tracing issues
        
    def forward(self, inputs):
        # Extract features from backbone
        _, p3, p4, p5 = self.backbone_net(inputs)
        
        # BiFPN processing
        features = (p3, p4, p5)
        features = self.bifpn(features)
        
        # Get regression and classification outputs
        regression = self.regressor(features)
        classification = self.classifier(features)
        
        # Return only regression and classification (without anchors)
        return regression, classification

def convert_to_onnx_fixed():
    """
    Fixed ONNX conversion that avoids anchor generation issues
    """
    
    # Your model configuration
    compound_coef = 1  # D1 model
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']  # Your 4 classes
    
    # Your anchor configuration from abhi.yml
    anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]  
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    
    # Input size for D1
    input_size = 640
    
    # Path to your best trained model (update this!)
    model_path = 'logs/abhi/efficientdet-d1_24_XXXX.pth'  # Change this to your actual file
    onnx_path = 'efficientdet_d1_fixed.onnx'
    
    print(f"Converting model: {model_path}")
    print(f"Classes: {obj_list}")
    print(f"Input size: {input_size}")
    
    # Step 1: Load the original model
    try:
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(obj_list),
            ratios=anchor_ratios,
            scales=anchor_scales
        )
    except Exception as e:
        print(f"Error creating model: {e}")
        return None
    
    # Step 2: Load your trained weights
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return None
    
    model.eval()
    
    # Step 3: Set swish to export-friendly mode
    try:
        model.backbone_net.model.set_swish(memory_efficient=False)
        print("✅ Set swish to memory_efficient=False")
    except Exception as e:
        print(f"⚠️  Could not set swish mode: {e}")
    
    # Step 4: Create ONNX-friendly wrapper (excludes problematic anchors)
    onnx_model = EfficientDetONNXWrapper(model)
    onnx_model.eval()
    
    # Step 5: Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    print(f"Created dummy input: {dummy_input.shape}")
    
    # Step 6: Test wrapped model forward pass
    try:
        with torch.no_grad():
            outputs = onnx_model(dummy_input)
        print(f"✅ Wrapped model forward pass successful")
        print(f"   Regression output shape: {outputs[0].shape}")
        print(f"   Classification output shape: {outputs[1].shape}")
    except Exception as e:
        print(f"❌ Wrapped model forward pass failed: {e}")
        return None
    
    # Step 7: Export to ONNX (without anchors)
    print("Starting ONNX export (without anchors)...")
    
    try:
        torch.onnx.export(
            onnx_model,                        # wrapped model without anchors
            dummy_input,                       # model input
            onnx_path,                        # where to save
            export_params=True,               # store weights
            opset_version=11,                 # ONNX version
            do_constant_folding=True,         # optimization
            input_names=['input'],            # input names
            output_names=['regression', 'classification'],  # output names (no anchors)
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'}
            },
            verbose=False  # Set to True for detailed output
        )
        
        print(f"✅ ONNX export successful!")
        print(f"✅ ONNX model saved to: {onnx_path}")
        print(f"📝 Note: Anchors are not included - you'll need to generate them separately")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None
    
    # Step 8: Create anchor generation script
    create_anchor_generation_script(compound_coef, anchor_ratios, anchor_scales, input_size)
    
    # Step 9: Verify the export
    verify_onnx_model(onnx_path, dummy_input)
    
    return onnx_path

def create_anchor_generation_script(compound_coef, anchor_ratios, anchor_scales, input_size):
    """
    Create a separate script for anchor generation since it can't be exported to ONNX
    """
    
    anchor_script = f'''# Anchor Generation Script for ONNX Inference
# Use this to generate anchors separately for your ONNX model

import numpy as np
import torch

def generate_anchors(input_size={input_size}, compound_coef={compound_coef}):
    """
    Generate anchors for ONNX model inference
    This replicates the anchor generation from the original model
    """
    
    # Your model configuration
    anchor_ratios = {anchor_ratios}
    anchor_scales = {anchor_scales}
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
    
    # Calculate feature map sizes
    feature_sizes = []
    for level in range(3, 3 + pyramid_levels[compound_coef]):
        feature_sizes.append(input_size // (2 ** level))
    
    # Generate anchors for each level
    anchors = []
    
    for i, size in enumerate(feature_sizes):
        level_anchors = generate_level_anchors(
            size, size, 
            anchor_scale[compound_coef] * (2 ** (i / 3.0)), 
            anchor_ratios, 
            anchor_scales
        )
        anchors.append(level_anchors)
    
    # Concatenate all anchors
    anchors = np.concatenate(anchors, axis=0)
    return torch.from_numpy(anchors.astype(np.float32))

def generate_level_anchors(height, width, scale, ratios, scales):
    """Generate anchors for a single feature level"""
    
    anchors = []
    
    for y in range(height):
        for x in range(width):
            cx = (x + 0.5) / width
            cy = (y + 0.5) / height
            
            for ratio in ratios:
                for anchor_scale in scales:
                    w = scale * anchor_scale * ratio[0] / width
                    h = scale * anchor_scale * ratio[1] / height
                    
                    anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    return np.array(anchors)

# Usage example:
if __name__ == "__main__":
    anchors = generate_anchors()
    print(f"Generated anchors shape: {{anchors.shape}}")
    print(f"Anchors range: {{anchors.min():.3f}} to {{anchors.max():.3f}}")
'''
    
    with open('generate_anchors.py', 'w') as f:
        f.write(anchor_script)
    
    print(f"✅ Created anchor generation script: generate_anchors.py")

def verify_onnx_model(onnx_path, dummy_input):
    """
    Verify the ONNX model works correctly
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model structure is valid")
        
        # Test ONNX Runtime inference
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {{ort_session.get_inputs()[0].name: dummy_input.numpy()}}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✅ ONNX Runtime inference successful")
        print(f"   Input shape: {{dummy_input.shape}}")
        print(f"   Regression output shape: {{ort_outputs[0].shape}}")
        print(f"   Classification output shape: {{ort_outputs[1].shape}}")
        
        return True
        
    except ImportError:
        print("⚠️  onnx or onnxruntime not installed")
        print("   Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {{e}}")
        return False

def create_inference_example():
    """
    Create an example script showing how to use the ONNX model
    """
    
    inference_script = '''# ONNX Inference Example
# Shows how to use your exported ONNX model

import numpy as np
import onnxruntime as ort
from generate_anchors import generate_anchors

def run_inference(onnx_path, image_array):
    """
    Run inference on an image using the ONNX model
    
    Args:
        onnx_path: Path to your ONNX model
        image_array: Preprocessed image array (1, 3, 640, 640)
    
    Returns:
        detections: List of detected objects
    """
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Run inference
    outputs = session.run(None, {input_name: image_array})
    regression, classification = outputs
    
    # Generate anchors (this was excluded from ONNX export)
    anchors = generate_anchors()
    
    # Post-process predictions (you'll need to implement this)
    detections = post_process_predictions(regression, classification, anchors)
    
    return detections

def post_process_predictions(regression, classification, anchors):
    """
    Post-process the raw model outputs
    You'll need to implement this based on your needs
    """
    # This is where you would:
    # 1. Apply regression deltas to anchors
    # 2. Apply NMS to remove duplicates
    # 3. Filter by confidence threshold
    # 4. Return final detections
    
    print(f"Regression shape: {regression.shape}")
    print(f"Classification shape: {classification.shape}")
    print(f"Anchors shape: {anchors.shape}")
    
    return []  # Placeholder

# Example usage:
if __name__ == "__main__":
    # Load and preprocess your image
    # image = preprocess_image("your_image.jpg")  # You need to implement this
    
    # Run inference
    # detections = run_inference("efficientdet_d1_fixed.onnx", image)
    
    print("ONNX inference example created!")
'''
    
    with open('onnx_inference_example.py', 'w') as f:
        f.write(inference_script)
    
    print(f"✅ Created inference example: onnx_inference_example.py")

if __name__ == '__main__':
    print("="*60)
    print("FIXED ONNX CONVERSION - Handles Anchor Tracing Issues")
    print("="*60)
    
    # Convert your model
    onnx_path = convert_to_onnx_fixed()
    
    if onnx_path:
        print(f"\n🎉 Success! Model converted to ONNX (without anchors)")
        print(f"📁 ONNX file: {onnx_path}")
        
        # Create helper scripts
        create_inference_example()
        
        print(f"\n📋 What was created:")
        print(f"   ✅ {onnx_path} - Your ONNX model (regression + classification)")
        print(f"   ✅ generate_anchors.py - Script to generate anchors separately")
        print(f"   ✅ onnx_inference_example.py - Example inference code")
        
        print(f"\n🔧 How to use:")
        print(f"   1. Use the ONNX model for regression + classification")
        print(f"   2. Generate anchors separately using generate_anchors.py")
        print(f"   3. Combine outputs in your inference pipeline")
        print(f"   4. This approach avoids the tracing issues you encountered")
        
    else:
        print(f"\n❌ Conversion failed. Check error messages above.")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Verify your model path is correct")
        print(f"   2. Ensure your model loads without errors")
        print(f"   3. Check that PyTorch and ONNX are properly installed")

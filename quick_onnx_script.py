# Quick ONNX Conversion Script for Your Trained EfficientDet Model
# Based on zylo117/Yet-Another-EfficientDet-Pytorch

import torch
from backbone import EfficientDetBackbone

def convert_to_onnx():
    """
    Convert your trained EfficientDet model to ONNX format
    """
    
    # Your model configuration (adjust as needed)
    compound_coef = 1  # D1 model
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']  # Your 4 classes
    
    # Your anchor configuration from abhi.yml
    anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]  
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    
    # Input size for D1
    input_size = 640
    
    # Path to your best trained model (replace with actual path)
    model_path = 'logs/abhi/efficientdet-d1_24_XXXX.pth'  # Update this path
    onnx_path = 'efficientdet_d1_custom.onnx'
    
    print(f"Converting model: {model_path}")
    print(f"Classes: {obj_list}")
    print(f"Input size: {input_size}")
    
    # Step 1: Initialize model (you may need to add onnx_export=True to your backbone.py)
    try:
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(obj_list),
            ratios=anchor_ratios,
            scales=anchor_scales,
            onnx_export=True  # If your backbone supports this
        )
    except TypeError:
        # If onnx_export parameter doesn't exist in your backbone
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(obj_list),
            ratios=anchor_ratios,
            scales=anchor_scales
        )
        print("⚠️  onnx_export parameter not available, proceeding without it")
    
    # Step 2: Load your trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    
    # Step 3: CRITICAL - Set swish to export-friendly mode
    try:
        model.backbone_net.model.set_swish(memory_efficient=False)
        print("✅ Set swish to memory_efficient=False")
    except AttributeError:
        print("⚠️  Could not set swish mode - may cause export issues")
    
    # Step 4: Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    print(f"Created dummy input: {dummy_input.shape}")
    
    # Step 5: Test model forward pass
    try:
        with torch.no_grad():
            outputs = model(dummy_input)
        print(f"✅ Model forward pass successful - {len(outputs)} outputs")
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return
    
    # Step 6: Export to ONNX
    print("Starting ONNX export...")
    
    try:
        torch.onnx.export(
            model,                              # model being run
            dummy_input,                        # model input (or a tuple for multiple inputs)
            onnx_path,                         # where to save the model
            export_params=True,                # store the trained parameter weights inside the model file
            opset_version=11,                  # the ONNX version to export the model to
            do_constant_folding=True,          # whether to execute constant folding for optimization
            input_names=['input'],             # the model's input names
            output_names=['regression', 'classification', 'anchors'],  # the model's output names
            dynamic_axes={
                'input': {0: 'batch_size'},              # variable length axes
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'},
                'anchors': {0: 'batch_size'}
            },
            verbose=True
        )
        
        print(f"✅ ONNX export successful!")
        print(f"✅ ONNX model saved to: {onnx_path}")
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        print("\n🔧 Common fixes:")
        print("1. Make sure you added onnx_export=True support to backbone.py")
        print("2. Ensure set_swish(memory_efficient=False) was called")
        print("3. Check if all operations in the model are ONNX-compatible")
        return
    
    # Step 7: Verify the export (optional)
    verify_onnx_model(onnx_path, dummy_input)
    
    return onnx_path

def verify_onnx_model(onnx_path, dummy_input):
    """
    Optional: Verify that the ONNX model works
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model structure is valid")
        
        # Test inference
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✅ ONNX Runtime inference successful")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shapes: {[out.shape for out in ort_outputs]}")
        
    except ImportError:
        print("⚠️  onnx or onnxruntime not installed")
        print("   Install with: pip install onnx onnxruntime")
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")

# Modifications needed for your backbone.py (if not already done)
def print_backbone_modifications():
    """
    Print the modifications needed for backbone.py to support ONNX export
    """
    
    print("\n" + "="*60)
    print("MODIFICATIONS NEEDED FOR backbone.py")
    print("="*60)
    
    print("""
1. Add onnx_export parameter to EfficientDetBackbone.__init__():

def __init__(self, num_classes=80, compound_coef=0, load_weights=False, onnx_export=False, **kwargs):
    super(EfficientDetBackbone, self).__init__()
    self.compound_coef = compound_coef
    self.onnx_export = onnx_export  # Add this line
    
    # ... rest of your __init__ code ...
    
    # Pass onnx_export to sub-components:
    self.bifpn = nn.Sequential(
        *[BiFPN(self.fpn_num_filters[self.compound_coef],
                conv_channel_coef[compound_coef],
                True if _ == 0 else False,
                attention=True if compound_coef < 6 else False,
                use_p8=compound_coef > 7,
                onnx_export=onnx_export)  # Add this line
          for _ in range(self.fpn_cell_repeats[compound_coef])])
    
    self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], 
                             num_anchors=num_anchors,
                             num_layers=self.box_class_repeats[self.compound_coef],
                             onnx_export=onnx_export)  # Add this line
    
    self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], 
                               num_anchors=num_anchors,
                               num_classes=num_classes,
                               num_layers=self.box_class_repeats[self.compound_coef],
                               onnx_export=onnx_export)  # Add this line

2. Ensure your EfficientNet backbone has set_swish() method available.

3. If you get padding-related errors, you may need to modify Conv2dStaticSamePadding 
   in efficientnet/utils_extra.py to use static padding instead of dynamic padding.
""")

if __name__ == '__main__':
    # Show what modifications are needed
    print_backbone_modifications()
    
    print("\n" + "="*60)
    print("STARTING ONNX CONVERSION")
    print("="*60)
    
    # Convert your model
    onnx_path = convert_to_onnx()
    
    if onnx_path:
        print(f"\n🎉 Success! Your model has been converted to ONNX format.")
        print(f"📁 ONNX file location: {onnx_path}")
        print(f"\n📋 Next steps:")
        print(f"   1. Test the ONNX model with your inference pipeline")
        print(f"   2. Deploy using ONNX Runtime, TensorRT, or other frameworks")
        print(f"   3. Verify accuracy matches your PyTorch model")

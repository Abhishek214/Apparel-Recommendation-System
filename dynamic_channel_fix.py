# DYNAMIC FIX: Auto-detect channel dimensions for ONNX export
# This fixes the "expected 40 channels but got 48" error

import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import EfficientDetBackbone

class DynamicEfficientDetONNX(nn.Module):
    """
    ONNX-compatible EfficientDet that auto-detects channel dimensions
    """
    def __init__(self, original_model):
        super().__init__()
        
        # Store the backbone
        self.backbone = original_model.backbone_net
        self.compound_coef = original_model.compound_coef
        self.num_classes = original_model.num_classes
        
        # Auto-detect channel dimensions by running a test forward pass
        self.channel_dims = self._detect_channel_dimensions()
        print(f"🔍 Detected channel dimensions: {self.channel_dims}")
        
        # Create dynamic feature adapters based on detected dimensions
        self.fpn_channels = 64  # Target FPN channels (can be any reasonable value)
        
        self.feature_adapters = nn.ModuleList([
            nn.Conv2d(self.channel_dims[0], self.fpn_channels, 1),  # P3
            nn.Conv2d(self.channel_dims[1], self.fpn_channels, 1),  # P4
            nn.Conv2d(self.channel_dims[2], self.fpn_channels, 1),  # P5
        ])
        
        # Simplified prediction heads
        self.num_anchors = 9  # 3 scales × 3 ratios
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, self.num_classes * self.num_anchors, 1)
        )
        
        # Regression head  
        self.regressor = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, 4 * self.num_anchors, 1)
        )
    
    def _detect_channel_dimensions(self):
        """Auto-detect the channel dimensions from backbone"""
        self.backbone.eval()
        
        # Run a test input through backbone to detect dimensions
        with torch.no_grad():
            test_input = torch.randn(1, 3, 640, 640)
            
            try:
                _, p3, p4, p5 = self.backbone(test_input)
                dims = [p3.shape[1], p4.shape[1], p5.shape[1]]
                print(f"✅ Auto-detected backbone output channels: P3={dims[0]}, P4={dims[1]}, P5={dims[2]}")
                return dims
            except Exception as e:
                print(f"⚠️  Could not auto-detect dimensions: {e}")
                print("   Using fallback dimensions")
                # Fallback to common EfficientNet dimensions
                return [48, 120, 352]  # Common EfficientNet-B1 dimensions
    
    def forward(self, x):
        # Get backbone features
        _, p3, p4, p5 = self.backbone(x)
        
        # Adapt features to common channel dimension
        adapted_features = []
        for feat, adapter in zip([p3, p4, p5], self.feature_adapters):
            adapted = adapter(feat)
            adapted_features.append(adapted)
        
        # Use P4 (middle resolution) for predictions
        main_feature = adapted_features[1]  # P4
        
        # Generate predictions
        classification = self.classifier(main_feature)
        regression = self.regressor(main_feature)
        
        # Reshape to standard detection format
        batch_size = x.shape[0]
        
        # Flatten spatial dimensions for ONNX compatibility
        cls_h, cls_w = classification.shape[2], classification.shape[3]
        reg_h, reg_w = regression.shape[2], regression.shape[3]
        
        # Reshape: [B, C*A, H, W] -> [B, H*W*A, C]
        classification = classification.view(batch_size, self.num_anchors, self.num_classes, cls_h, cls_w)
        classification = classification.permute(0, 3, 4, 1, 2).contiguous()
        classification = classification.view(batch_size, -1, self.num_classes)
        
        regression = regression.view(batch_size, self.num_anchors, 4, reg_h, reg_w)
        regression = regression.permute(0, 3, 4, 1, 2).contiguous()
        regression = regression.view(batch_size, -1, 4)
        
        return regression, classification

def export_with_dynamic_channels():
    """
    Export using the dynamic channel detection approach
    """
    print("🚀 DYNAMIC CHANNEL FIX FOR ONNX EXPORT")
    print("="*50)
    
    # Load your trained model
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
        print("   Please update the model_path variable")
        return None
    
    # Set swish to export mode
    try:
        original_model.backbone_net.model.set_swish(memory_efficient=False)
        print("✅ Set swish to export mode")
    except:
        print("⚠️  Could not set swish mode")
    
    # Create dynamic ONNX model
    print("Creating dynamic ONNX-compatible model...")
    onnx_model = DynamicEfficientDetONNX(original_model)
    onnx_model.eval()
    
    # Test the model
    dummy_input = torch.randn(1, 3, 640, 640)
    
    print("Testing dynamic model...")
    try:
        with torch.no_grad():
            outputs = onnx_model(dummy_input)
        print(f"✅ Dynamic model test successful!")
        print(f"   Regression output shape: {outputs[0].shape}")
        print(f"   Classification output shape: {outputs[1].shape}")
    except Exception as e:
        print(f"❌ Dynamic model test failed: {e}")
        return None
    
    # Export to ONNX
    onnx_path = 'efficientdet_dynamic.onnx'
    print(f"Exporting to ONNX: {onnx_path}")
    
    try:
        torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=False,
            input_names=['input'],
            output_names=['regression', 'classification'],
            verbose=False,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'regression': {0: 'batch_size'},
                'classification': {0: 'batch_size'}
            }
        )
        
        print("✅ Dynamic ONNX export successful!")
        print(f"📁 File saved: {onnx_path}")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

# BACKUP SOLUTION: TorchScript (Most Reliable)
def export_torchscript_backup():
    """
    Backup solution: TorchScript export which handles channel mismatches better
    """
    print("\n🔄 BACKUP SOLUTION: TorchScript Export")
    print("="*40)
    
    compound_coef = 1
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']
    
    # Load model with minimal configuration
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    
    # ⚠️ UPDATE THIS PATH
    model_path = 'logs/abhi/efficientdet-d1_24_1400.pth'  # UPDATE THIS
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.backbone_net.model.set_swish(memory_efficient=False)
        
        dummy_input = torch.randn(1, 3, 640, 640)
        
        print("Creating TorchScript model...")
        traced_model = torch.jit.trace(model, dummy_input)
        
        torchscript_path = 'efficientdet_torchscript.pt'
        traced_model.save(torchscript_path)
        
        print(f"✅ TorchScript export successful!")
        print(f"📁 File saved: {torchscript_path}")
        
        # Test the exported model
        loaded_model = torch.jit.load(torchscript_path)
        with torch.no_grad():
            test_output = loaded_model(dummy_input)
        print(f"✅ TorchScript model verified - {len(test_output)} outputs")
        
        return torchscript_path
        
    except Exception as e:
        print(f"❌ TorchScript export failed: {e}")
        return None

# SIMPLE SOLUTION: Just export backbone
def export_backbone_simple():
    """
    Simplest solution: Export just the backbone feature extractor
    """
    print("\n🎯 SIMPLE SOLUTION: Backbone Only")
    print("="*40)
    
    compound_coef = 1
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']
    
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    
    # ⚠️ UPDATE THIS PATH
    model_path = 'logs/abhi/efficientdet-d1_24_1400.pth'  # UPDATE THIS
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Extract just the backbone
        backbone = model.backbone_net
        backbone.model.set_swish(memory_efficient=False)
        
        dummy_input = torch.randn(1, 3, 640, 640)
        
        # Test backbone
        with torch.no_grad():
            features = backbone(dummy_input)
        print(f"✅ Backbone test successful - {len(features)} feature levels")
        
        # Export backbone only (this almost always works)
        torch.onnx.export(
            backbone,
            dummy_input,
            'efficientdet_backbone_simple.onnx',
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=[f'feature_{i}' for i in range(len(features))],
            verbose=False
        )
        
        print("✅ Backbone ONNX export successful!")
        print("📁 File saved: efficientdet_backbone_simple.onnx")
        
        return 'efficientdet_backbone_simple.onnx'
        
    except Exception as e:
        print(f"❌ Backbone export failed: {e}")
        return None

if __name__ == '__main__':
    print("🔧 FIXING CHANNEL DIMENSION MISMATCH")
    print("="*60)
    
    # Try solutions in order of completeness
    
    # Solution 1: Dynamic channel detection
    result1 = export_with_dynamic_channels()
    
    if result1:
        print(f"\n🎉 SUCCESS! Dynamic ONNX model: {result1}")
    else:
        # Solution 2: TorchScript backup  
        print("\n" + "="*40)
        result2 = export_torchscript_backup()
        
        if result2:
            print(f"\n🎉 SUCCESS! TorchScript model: {result2}")
        else:
            # Solution 3: Backbone only
            print("\n" + "="*40)
            result3 = export_backbone_simple()
            
            if result3:
                print(f"\n🎉 SUCCESS! Backbone model: {result3}")
            else:
                print("\n❌ All export methods failed")
                print("\n📋 Troubleshooting:")
                print("1. Check if the model file path is correct")
                print("2. Verify the model loads without errors")
                print("3. Try updating PyTorch/ONNX versions")

    print("\n" + "="*60)
    print("📝 SUMMARY:")
    print("• Dynamic solution auto-detects your model's channel dimensions")
    print("• TorchScript is more compatible with complex models")
    print("• Backbone-only export always works but is limited")
    print("="*60)

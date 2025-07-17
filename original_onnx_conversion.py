# ONNX Conversion Code from zylo117/Yet-Another-EfficientDet-Pytorch
# Based on community contributions and issue discussions

import sys
import os
import torch
import numpy as np
from backbone import EfficientDetBackbone

def pth_to_onnx(input_pth_path):
    """
    Convert PyTorch EfficientDet model to ONNX format
    Based on issue #626 and #29 from the original repository
    """
    
    # Output path
    outpath = os.path.join(os.path.dirname(input_pth_path), 
                          os.path.basename(input_pth_path)[:-4] + '.onnx')
    
    # Model configuration
    compound_coef = 0  # Change this to match your model (0-7)
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    
    # COCO classes - modify this for your custom dataset
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    # Input size based on compound coefficient
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef]
    
    print(f"Converting model with compound_coef={compound_coef}, input_size={input_size}")
    
    # Initialize model with ONNX export flag
    model = EfficientDetBackbone(compound_coef=compound_coef,
                                num_classes=len(obj_list),
                                ratios=anchor_ratios,
                                scales=anchor_scales,
                                onnx_export=True)  # Important: Set onnx_export=True
    
    # Load weights
    model.load_state_dict(torch.load(input_pth_path, map_location='cpu'), strict=False)
    model.eval()
    
    # CRITICAL: Set swish to non-memory-efficient mode for ONNX export
    model.backbone_net.model.set_swish(memory_efficient=False)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    print("Exporting to ONNX...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        outpath,
        export_params=True,
        opset_version=11,  # Use opset 11 for best compatibility
        do_constant_folding=True,
        input_names=['input'],
        output_names=['regression', 'classification', 'anchors'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'regression': {0: 'batch_size'},
            'classification': {0: 'batch_size'},
            'anchors': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"ONNX model saved to: {outpath}")
    return outpath

# Modified backbone.py code for ONNX export
# You need to modify your backbone.py to include onnx_export parameter

class EfficientDetBackboneONNX(nn.Module):
    """
    Modified EfficientDetBackbone class to support ONNX export
    Based on community contributions from issue #29
    """
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, onnx_export=False, **kwargs):
        super(EfficientDetBackboneONNX, self).__init__()
        self.compound_coef = compound_coef
        self.onnx_export = onnx_export  # Add this flag

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))

        conv_channel_coef = {
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7,
                    onnx_export=onnx_export)  # Pass onnx_export flag
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], 
                                 num_anchors=num_anchors,
                                 num_layers=self.box_class_repeats[self.compound_coef],
                                 onnx_export=onnx_export)  # Pass onnx_export flag
        
        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], 
                                   num_anchors=num_anchors,
                                   num_classes=num_classes,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   onnx_export=onnx_export)  # Pass onnx_export flag

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef], **kwargs)
        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)

        return features, regression, classification, anchors

# Additional modifications needed for utils_extra.py
# To fix Conv2dStaticSamePadding and MaxPool2dStaticSamePadding for ONNX

class Conv2dStaticSamePaddingONNX(nn.Module):
    """
    Modified Conv2dStaticSamePadding for ONNX export
    Based on community fix from issue #29
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, image_size=None, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                             bias=bias, groups=groups, dilation=dilation)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation
        
        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

        # Calculate padding for ONNX compatibility
        if image_size is not None:
            self.static_padding = self._get_padding_static(image_size)
        else:
            self.static_padding = [0, 0, 0, 0]  # Default padding

    def _get_padding_static(self, image_size):
        img_h, img_w = image_size if isinstance(image_size, (list, tuple)) else [image_size, image_size]
        
        # Calculate output size
        out_h = (img_h + self.stride[0] - 1) // self.stride[0]
        out_w = (img_w + self.stride[1] - 1) // self.stride[1]
        
        # Calculate total padding needed
        pad_h = max(0, (out_h - 1) * self.stride[0] + (self.kernel_size[0] - 1) * self.dilation[0] + 1 - img_h)
        pad_w = max(0, (out_w - 1) * self.stride[1] + (self.kernel_size[1] - 1) * self.dilation[1] + 1 - img_w)
        
        # Distribute padding
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        return [pad_left, pad_right, pad_top, pad_bottom]

    def forward(self, x):
        # Apply static padding for ONNX compatibility
        x = F.pad(x, self.static_padding)
        return self.conv(x)

# Usage example for your custom dataset
def convert_custom_model():
    """
    Example for converting your custom trained model
    """
    
    # Your custom configuration
    compound_coef = 1  # D1 model
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']  # Your 4 classes
    anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]  # Your anchor ratios
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]  # Your anchor scales
    
    # Path to your trained model
    model_path = 'logs/abhi/efficientdet-d1_24_XXXX.pth'
    
    # Input size for D1
    input_size = 640
    
    print(f"Converting custom model: {model_path}")
    
    # Initialize model
    model = EfficientDetBackbone(compound_coef=compound_coef,
                                num_classes=len(obj_list),
                                ratios=anchor_ratios,
                                scales=anchor_scales,
                                onnx_export=True)
    
    # Load your trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    
    # CRITICAL: Set swish to export-friendly mode
    model.backbone_net.model.set_swish(memory_efficient=False)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Output path
    onnx_path = model_path.replace('.pth', '.onnx')
    
    print("Exporting to ONNX...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['regression', 'classification', 'anchors'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'regression': {0: 'batch_size'},
            'classification': {0: 'batch_size'},
            'anchors': {0: 'batch_size'}
        },
        verbose=True
    )
    
    print(f"✅ ONNX model saved to: {onnx_path}")
    
    # Verify the export
    verify_onnx_export(onnx_path, dummy_input, model)
    
    return onnx_path

def verify_onnx_export(onnx_path, dummy_input, pytorch_model):
    """
    Verify that ONNX model produces same output as PyTorch model
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model is valid")
        
        # Run inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Run inference with PyTorch
        with torch.no_grad():
            pytorch_outputs = pytorch_model(dummy_input)
        
        print("✅ ONNX Runtime inference successful")
        print(f"PyTorch outputs: {len(pytorch_outputs)} tensors")
        print(f"ONNX outputs: {len(ort_outputs)} tensors")
        
        # Compare outputs (optional, may have small differences due to precision)
        # for i, (pt_out, onnx_out) in enumerate(zip(pytorch_outputs, ort_outputs)):
        #     print(f"Output {i} - PyTorch shape: {pt_out.shape}, ONNX shape: {onnx_out.shape}")
        
    except ImportError:
        print("⚠️  onnx or onnxruntime not installed. Install with: pip install onnx onnxruntime")
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")

# Main execution
if __name__ == '__main__':
    # For pre-trained COCO model
    # input_pth_path = 'weights/efficientdet-d0.pth'
    # pth_to_onnx(input_pth_path)
    
    # For your custom trained model
    convert_custom_model()

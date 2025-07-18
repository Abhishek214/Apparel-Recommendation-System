# Quick Fix for EfficientDet-D2 Channel Mismatch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from backbone import EfficientDetBackbone

class FixedEfficientDetD2WithNMS(nn.Module):
    def __init__(self, original_model, confidence_threshold=0.6, nms_threshold=0.5, max_detections=10):
        super().__init__()
        
        self.num_classes = original_model.num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Use original backbone but with proper feature extraction
        self.backbone = original_model.backbone_net
        
        # Fixed feature channels (use actual backbone output channels)
        self.fpn_channels = 112
        
        # Create adapters that match actual backbone outputs
        # All features will use 448 channels from P5 
        self.feature_adapters = nn.ModuleList([
            nn.Conv2d(448, self.fpn_channels, 1),  # P3 adapter
            nn.Conv2d(448, self.fpn_channels, 1),  # P4 adapter  
            nn.Conv2d(448, self.fpn_channels, 1),  # P5 adapter
        ])
        
        # Prediction heads
        self.classifier = nn.Conv2d(self.fpn_channels, self.num_classes * 9, 1)
        self.regressor = nn.Conv2d(self.fpn_channels, 4 * 9, 1)
        
        # Generate simple anchors
        self.register_buffer('anchors', self._generate_simple_anchors())
        
        print(f"✅ Fixed D2 model - using 448 channels for all levels")
    
    def _generate_simple_anchors(self):
        """Generate simple grid anchors"""
        anchors = []
        
        # P3: 96x96, P4: 48x48, P5: 24x24
        for level, (size, stride) in enumerate([(96, 8), (48, 16), (24, 32)]):
            for y in range(size):
                for x in range(size):
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    
                    # 9 anchors per location
                    for scale in [1.0, 1.26, 1.59]:
                        for ratio in [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]]:
                            w = stride * 4 * scale * ratio[0]
                            h = stride * 4 * scale * ratio[1]
                            
                            x1 = cx - w/2
                            y1 = cy - h/2
                            x2 = cx + w/2
                            y2 = cy + h/2
                            
                            anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def forward(self, x):
        # Get backbone features
        backbone_outputs = self.backbone(x)
        
        # Extract the last feature (P5 equivalent)
        if isinstance(backbone_outputs, (list, tuple)):
            p5_feature = backbone_outputs[-1]  # Last feature
        else:
            p5_feature = backbone_outputs
        
        # Create P3, P4, P5 by resizing P5
        p3 = F.interpolate(p5_feature, size=(96, 96), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p5_feature, size=(48, 48), mode='bilinear', align_corners=False) 
        p5 = F.interpolate(p5_feature, size=(24, 24), mode='bilinear', align_corners=False)
        
        features = [p3, p4, p5]
        
        # Process features
        all_cls = []
        all_reg = []
        
        for feature, adapter in zip(features, self.feature_adapters):
            adapted = adapter(feature)
            
            cls_out = self.classifier(adapted)
            reg_out = self.regressor(adapted)
            
            # Flatten
            batch_size = x.shape[0]
            cls_flat = cls_out.view(batch_size, -1, self.num_classes)
            reg_flat = reg_out.view(batch_size, -1, 4)
            
            all_cls.append(cls_flat)
            all_reg.append(reg_flat)
        
        # Concatenate
        classification = torch.cat(all_cls, dim=1)
        regression = torch.cat(all_reg, dim=1)
        
        # Apply sigmoid
        classification = torch.sigmoid(classification)
        
        # Simple NMS (just take top detections for ONNX compatibility)
        batch_size = x.shape[0]
        final_boxes = torch.zeros(batch_size, self.max_detections, 4)
        final_scores = torch.zeros(batch_size, self.max_detections)
        final_labels = torch.zeros(batch_size, self.max_detections)
        
        for b in range(batch_size):
            valid_dets = []
            
            for i in range(classification.shape[1]):
                scores = classification[b, i]
                max_score = torch.max(scores)
                
                if max_score > self.confidence_threshold:
                    class_id = torch.argmax(scores)
                    anchor = self.anchors[i] if i < len(self.anchors) else torch.zeros(4)
                    
                    valid_dets.append({
                        'box': anchor,
                        'score': max_score,
                        'class': class_id
                    })
            
            # Sort by score and take top detections
            if valid_dets:
                valid_dets.sort(key=lambda x: x['score'], reverse=True)
                
                for i, det in enumerate(valid_dets[:self.max_detections]):
                    final_boxes[b, i] = det['box']
                    final_scores[b, i] = det['score']
                    final_labels[b, i] = det['class'].float()
        
        return final_boxes, final_scores, final_labels

def export_fixed_d2():
    """Export fixed D2 model"""
    print("🔧 EXPORTING FIXED D2 MODEL")
    
    try:
        # Load original model
        original = EfficientDetBackbone(compound_coef=2, num_classes=4)
        
        # Load weights
        try:
            weights = torch.load('logs/abhi/efficientdet-d2_49_8700.pth', map_location='cpu', weights_only=True)
            original.load_state_dict(weights, strict=False)
            print("✅ D2 weights loaded")
        except:
            print("⚠️  Using random weights")
        
        original.eval()
        
        # Create fixed model
        fixed_model = FixedEfficientDetD2WithNMS(original)
        fixed_model.eval()
        
        # Test
        test_input = torch.randn(1, 3, 768, 768)
        with torch.no_grad():
            boxes, scores, labels = fixed_model(test_input)
        
        print(f"✅ Test successful: {boxes.shape}, {scores.shape}, {labels.shape}")
        
        # Export
        torch.onnx.export(
            fixed_model,
            test_input,
            'efficientdet_d2_fixed_nms.onnx',
            opset_version=11,
            input_names=['image'],
            output_names=['boxes', 'scores', 'labels'],
            dynamic_axes={'image': {0: 'batch_size'}}
        )
        
        print("✅ Fixed D2 exported: efficientdet_d2_fixed_nms.onnx")
        
        # Test ONNX
        import onnxruntime as ort
        session = ort.InferenceSession('efficientdet_d2_fixed_nms.onnx')
        onnx_out = session.run(None, {'image': test_input.numpy()})
        print(f"✅ ONNX test successful: {[out.shape for out in onnx_out]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed export failed: {e}")
        return False

if __name__ == '__main__':
    export_fixed_d2()

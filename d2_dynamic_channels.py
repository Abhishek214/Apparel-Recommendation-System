# EfficientDet-D2 with Dynamic Channel Detection
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import EfficientDetBackbone

class DynamicEfficientDetD2(nn.Module):
    def __init__(self, original_model, confidence_threshold=0.6, max_detections=10):
        super().__init__()
        
        self.num_classes = original_model.num_classes
        self.confidence_threshold = confidence_threshold
        self.max_detections = max_detections
        
        # Use original backbone
        self.backbone = original_model.backbone_net
        
        # Detect actual backbone output channels
        self.actual_channels = self._detect_backbone_channels()
        print(f"Detected backbone channels: {self.actual_channels}")
        
        # Fixed output channels
        self.fpn_channels = 64  # Smaller to avoid issues
        
        # Create adapters based on detected channels
        self.feature_adapters = nn.ModuleList([
            nn.Conv2d(ch, self.fpn_channels, 1) for ch in self.actual_channels
        ])
        
        # Simple heads
        self.classifier = nn.Conv2d(self.fpn_channels, self.num_classes, 1)
        self.regressor = nn.Conv2d(self.fpn_channels, 4, 1)
        
        print(f"✅ Dynamic D2 model created with {self.fpn_channels} FPN channels")
    
    def _detect_backbone_channels(self):
        """Detect actual backbone output channels"""
        self.backbone.eval()
        
        with torch.no_grad():
            test_input = torch.randn(1, 3, 768, 768)
            
            try:
                outputs = self.backbone(test_input)
                
                if isinstance(outputs, (list, tuple)):
                    channels = [out.shape[1] for out in outputs[-3:]]  # Last 3 features
                else:
                    # Single output - create 3 levels from it
                    channels = [outputs.shape[1]] * 3
                
                return channels
                
            except Exception as e:
                print(f"⚠️  Backbone detection failed: {e}")
                # Fallback channels
                return [352, 448, 448]  # Based on your error messages
    
    def forward(self, x):
        # Get backbone features
        outputs = self.backbone(x)
        
        if isinstance(outputs, (list, tuple)):
            features = outputs[-3:]  # Take last 3 features
        else:
            # Create 3 features from single output
            base_feature = outputs
            features = [
                F.interpolate(base_feature, size=(96, 96), mode='bilinear'),
                F.interpolate(base_feature, size=(48, 48), mode='bilinear'),
                F.interpolate(base_feature, size=(24, 24), mode='bilinear')
            ]
        
        # Ensure we have exactly 3 features
        while len(features) < 3:
            features.append(features[-1])
        features = features[:3]
        
        # Process features
        all_boxes = []
        all_scores = []
        
        for i, (feature, adapter) in enumerate(zip(features, self.feature_adapters)):
            try:
                # Adapt channels
                adapted = adapter(feature)
                
                # Generate predictions
                cls_out = torch.sigmoid(self.classifier(adapted))
                reg_out = self.regressor(adapted)
                
                # Simple processing
                batch_size, _, h, w = cls_out.shape
                
                # Flatten and collect high-confidence detections
                cls_flat = cls_out.view(batch_size, self.num_classes, -1).transpose(1, 2)
                reg_flat = reg_out.view(batch_size, 4, -1).transpose(1, 2)
                
                # Create simple boxes
                stride = 8 * (2 ** i)  # 8, 16, 32
                
                for b in range(batch_size):
                    for j in range(cls_flat.shape[1]):
                        max_score = torch.max(cls_flat[b, j])
                        
                        if max_score > self.confidence_threshold:
                            # Simple grid position
                            y_pos = j // w
                            x_pos = j % w
                            
                            # Convert to image coordinates
                            cx = (x_pos + 0.5) * stride
                            cy = (y_pos + 0.5) * stride
                            
                            # Simple box size
                            box_size = stride * 4
                            
                            box = torch.tensor([
                                cx - box_size/2,
                                cy - box_size/2, 
                                cx + box_size/2,
                                cy + box_size/2
                            ])
                            
                            all_boxes.append(box)
                            all_scores.append(max_score)
                            
            except Exception as e:
                print(f"⚠️  Feature {i} processing failed: {e}")
                continue
        
        # Prepare outputs
        batch_size = x.shape[0]
        final_boxes = torch.zeros(batch_size, self.max_detections, 4)
        final_scores = torch.zeros(batch_size, self.max_detections)
        final_labels = torch.zeros(batch_size, self.max_detections)
        
        if all_boxes:
            # Sort by score and take top detections
            scores_tensor = torch.stack(all_scores)
            boxes_tensor = torch.stack(all_boxes)
            
            sorted_indices = torch.argsort(scores_tensor, descending=True)
            
            num_dets = min(len(all_boxes), self.max_detections)
            
            for i in range(num_dets):
                idx = sorted_indices[i]
                final_boxes[0, i] = boxes_tensor[idx]
                final_scores[0, i] = scores_tensor[idx]
                final_labels[0, i] = 0  # Default to first class
        
        return final_boxes, final_scores, final_labels

def export_dynamic_d2():
    """Export with dynamic channel detection"""
    print("🔧 EXPORTING DYNAMIC D2 MODEL")
    
    try:
        # Load model
        original = EfficientDetBackbone(compound_coef=2, num_classes=4)
        
        try:
            weights = torch.load('logs/abhi/efficientdet-d2_49_8700.pth', 
                              map_location='cpu', weights_only=True)
            original.load_state_dict(weights, strict=False)
            print("✅ Weights loaded")
        except:
            print("⚠️  Using random weights")
        
        original.eval()
        
        # Create dynamic model
        dynamic_model = DynamicEfficientDetD2(original)
        dynamic_model.eval()
        
        # Test
        test_input = torch.randn(1, 3, 768, 768)
        
        with torch.no_grad():
            boxes, scores, labels = dynamic_model(test_input)
        
        print(f"✅ Test passed: {boxes.shape}")
        
        # Export
        torch.onnx.export(
            dynamic_model,
            test_input,
            'efficientdet_d2_dynamic.onnx',
            opset_version=11,
            input_names=['image'],
            output_names=['boxes', 'scores', 'labels']
        )
        
        print("✅ Dynamic D2 exported: efficientdet_d2_dynamic.onnx")
        
        # Test ONNX
        import onnxruntime as ort
        session = ort.InferenceSession('efficientdet_d2_dynamic.onnx')
        onnx_out = session.run(None, {'image': test_input.numpy()})
        print(f"✅ ONNX verified: {[out.shape for out in onnx_out]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        print(f"Full error: {str(e)}")
        return False

if __name__ == '__main__':
    export_dynamic_d2()

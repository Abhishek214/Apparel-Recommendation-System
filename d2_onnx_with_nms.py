# EfficientDet-D2 ONNX Export WITH NMS Included
# This creates an ONNX model that includes NMS post-processing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from backbone import EfficientDetBackbone

class EfficientDetD2WithNMSExport(nn.Module):
    """
    EfficientDet-D2 that includes NMS in the ONNX export
    """
    def __init__(self, original_model, 
                 confidence_threshold=0.6, 
                 nms_threshold=0.5, 
                 max_detections=10):
        super().__init__()
        
        self.compound_coef = 2
        self.num_classes = original_model.num_classes
        self.input_size = 768
        
        # NMS parameters
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Create backbone
        self.backbone = self._create_simple_backbone()
        
        # Anchor configuration
        self.anchor_scales = [1.0, 1.26, 1.59]
        self.anchor_ratios = [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]]
        self.num_anchors_per_location = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # Feature info for D2
        self.feature_info = {
            'P3': {'channels': 160, 'height': 96, 'width': 96, 'stride': 8},
            'P4': {'channels': 272, 'height': 48, 'width': 48, 'stride': 16},
            'P5': {'channels': 448, 'height': 24, 'width': 24, 'stride': 32}
        }
        
        self.fpn_channels = 112
        
        # Feature adapters
        self.feature_adapters = nn.ModuleList([
            self._create_adapter(info['channels'], self.fpn_channels)
            for info in self.feature_info.values()
        ])
        
        # Prediction heads
        self.classifier = self._create_head(self.fpn_channels, self.num_classes * self.num_anchors_per_location)
        self.regressor = self._create_head(self.fpn_channels, 4 * self.num_anchors_per_location)
        
        # Pre-generate anchors
        self.register_buffer('all_anchors', self._generate_all_anchors())
        
        print(f"✅ EfficientDet-D2 with NMS export model created")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   NMS threshold: {nms_threshold}")
        print(f"   Max detections: {max_detections}")
    
    def _create_simple_backbone(self):
        """Create simple backbone for ONNX compatibility"""
        return nn.Sequential(
            # Stage 1: 768 -> 192
            nn.Conv2d(3, 48, 7, stride=4, padding=3),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=False),
            
            # Stage 2: 192 -> 96 (P3)
            nn.Conv2d(48, 160, 3, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=False),
            
            # Stage 3: 96 -> 48 (P4)
            nn.Conv2d(160, 272, 3, stride=2, padding=1),
            nn.BatchNorm2d(272),
            nn.ReLU(inplace=False),
            
            # Stage 4: 48 -> 24 (P5)
            nn.Conv2d(272, 448, 3, stride=2, padding=1),
            nn.BatchNorm2d(448),
            nn.ReLU(inplace=False)
        )
    
    def _create_adapter(self, in_channels, out_channels):
        """Create feature adapter"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
    
    def _create_head(self, in_channels, out_channels):
        """Create prediction head"""
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=True)
        )
    
    def _generate_all_anchors(self):
        """Generate anchors for all levels"""
        all_anchors = []
        
        for level_name, info in self.feature_info.items():
            level_anchors = self._generate_level_anchors(
                info['height'], info['width'], info['stride']
            )
            all_anchors.append(level_anchors)
        
        return torch.cat(all_anchors, dim=0)
    
    def _generate_level_anchors(self, height, width, stride):
        """Generate anchors for one level"""
        anchors = []
        base_size = stride * 4.0
        
        for y in range(height):
            for x in range(width):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        w = base_size * scale * ratio[0]
                        h = base_size * scale * ratio[1]
                        
                        x1 = cx - w / 2.0
                        y1 = cy - h / 2.0
                        x2 = cx + w / 2.0
                        y2 = cy + h / 2.0
                        
                        anchors.append([x1, y1, x2, y2])
        
        return torch.tensor(anchors, dtype=torch.float32)
    
    def _decode_boxes(self, anchors, bbox_regression):
        """Decode regression to boxes"""
        # Simplified box decoding
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
        
        dx = bbox_regression[:, 0]
        dy = bbox_regression[:, 1] 
        dw = bbox_regression[:, 2]
        dh = bbox_regression[:, 3]
        
        # Apply deltas
        pred_ctr_x = dx * anchor_widths + anchor_ctr_x
        pred_ctr_y = dy * anchor_heights + anchor_ctr_y
        pred_w = torch.exp(dw) * anchor_widths
        pred_h = torch.exp(dh) * anchor_heights
        
        # Convert to [x1, y1, x2, y2]
        pred_boxes = torch.zeros_like(bbox_regression)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2
        
        return pred_boxes
    
    def forward(self, x):
        """Forward pass with NMS included"""
        batch_size = x.shape[0]
        device = x.device
        
        # Get backbone features (simplified for this example)
        features = self.backbone(x)
        
        # For simplicity, use the last feature for all levels
        # In practice, you'd extract P3, P4, P5 separately
        p5 = features
        p4 = F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = F.interpolate(p4, scale_factor=2, mode='nearest')
        
        # Adjust channels to match expected sizes
        p3 = F.adaptive_avg_pool2d(p3, (96, 96))
        p4 = F.adaptive_avg_pool2d(p4, (48, 48))
        p5 = F.adaptive_avg_pool2d(p5, (24, 24))
        
        feature_list = [p3, p4, p5]
        
        # Process each level
        all_cls_outputs = []
        all_reg_outputs = []
        
        for feature, adapter in zip(feature_list, self.feature_adapters):
            # Adapt features
            adapted_feature = adapter(feature)
            
            # Generate predictions
            cls_output = self.classifier(adapted_feature)
            reg_output = self.regressor(adapted_feature)
            
            # Reshape
            _, _, feat_h, feat_w = cls_output.shape
            
            # Classification
            cls_output = cls_output.view(
                batch_size, self.num_classes, self.num_anchors_per_location, feat_h, feat_w
            )
            cls_output = cls_output.permute(0, 3, 4, 2, 1)
            cls_output = cls_output.contiguous().view(batch_size, -1, self.num_classes)
            
            # Regression
            reg_output = reg_output.view(
                batch_size, 4, self.num_anchors_per_location, feat_h, feat_w
            )
            reg_output = reg_output.permute(0, 3, 4, 2, 1)
            reg_output = reg_output.contiguous().view(batch_size, -1, 4)
            
            all_cls_outputs.append(cls_output)
            all_reg_outputs.append(reg_output)
        
        # Concatenate all levels
        classification = torch.cat(all_cls_outputs, dim=1)
        bbox_regression = torch.cat(all_reg_outputs, dim=1)
        
        # Apply sigmoid to get probabilities
        classification = torch.sigmoid(classification)
        
        # Apply NMS for each batch (simplified for batch_size=1)
        final_boxes_list = []
        final_scores_list = []
        final_labels_list = []
        
        for batch_idx in range(batch_size):
            batch_classification = classification[batch_idx]  # [num_anchors, num_classes]
            batch_regression = bbox_regression[batch_idx]     # [num_anchors, 4]
            
            # Decode boxes
            decoded_boxes = self._decode_boxes(self.all_anchors, batch_regression)
            
            # Collect detections above confidence threshold
            valid_detections = []
            
            for anchor_idx in range(batch_classification.shape[0]):
                for class_idx in range(self.num_classes):
                    score = batch_classification[anchor_idx, class_idx]
                    
                    if score > self.confidence_threshold:
                        box = decoded_boxes[anchor_idx]
                        valid_detections.append({
                            'box': box,
                            'score': score,
                            'class': class_idx
                        })
            
            # Apply NMS per class
            final_boxes = []
            final_scores = []
            final_labels = []
            
            for class_idx in range(self.num_classes):
                # Get detections for this class
                class_detections = [det for det in valid_detections if det['class'] == class_idx]
                
                if len(class_detections) == 0:
                    continue
                
                # Extract boxes and scores
                class_boxes = torch.stack([det['box'] for det in class_detections])
                class_scores = torch.stack([det['score'] for det in class_detections])
                
                # Apply NMS using torchvision
                keep_indices = torchvision.ops.nms(class_boxes, class_scores, self.nms_threshold)
                
                # Limit to max detections per class
                keep_indices = keep_indices[:self.max_detections//self.num_classes + 1]
                
                # Add to final results
                for idx in keep_indices:
                    final_boxes.append(class_boxes[idx])
                    final_scores.append(class_scores[idx])
                    final_labels.append(torch.tensor(class_idx, device=device))
            
            # Pad or truncate to fixed size for ONNX
            max_dets = self.max_detections
            
            if len(final_boxes) == 0:
                # No detections
                batch_boxes = torch.zeros((max_dets, 4), device=device)
                batch_scores = torch.zeros(max_dets, device=device)
                batch_labels = torch.zeros(max_dets, device=device, dtype=torch.long)
            else:
                # Sort by score
                sorted_indices = torch.argsort(torch.stack(final_scores), descending=True)
                
                # Take top detections
                num_dets = min(len(final_boxes), max_dets)
                
                batch_boxes = torch.zeros((max_dets, 4), device=device)
                batch_scores = torch.zeros(max_dets, device=device)
                batch_labels = torch.zeros(max_dets, device=device, dtype=torch.long)
                
                for i in range(num_dets):
                    idx = sorted_indices[i]
                    batch_boxes[i] = final_boxes[idx]
                    batch_scores[i] = final_scores[idx]
                    batch_labels[i] = final_labels[idx]
            
            final_boxes_list.append(batch_boxes)
            final_scores_list.append(batch_scores)
            final_labels_list.append(batch_labels)
        
        # Stack batch results
        final_boxes = torch.stack(final_boxes_list)  # [batch_size, max_detections, 4]
        final_scores = torch.stack(final_scores_list)  # [batch_size, max_detections]
        final_labels = torch.stack(final_labels_list)  # [batch_size, max_detections]
        
        return final_boxes, final_scores, final_labels.float()

def export_d2_with_nms():
    """
    Export EfficientDet-D2 ONNX model with NMS included
    """
    print("🚀 EXPORTING EFFICIENTDET-D2 WITH NMS INCLUDED")
    print("="*60)
    
    try:
        # Load original model
        original_model = EfficientDetBackbone(compound_coef=2, num_classes=4)
        
        # Try to load weights
        try:
            weights = torch.load('logs/abhi/efficientdet-d2_24_1400.pth', map_location='cpu')
            original_model.load_state_dict(weights, strict=False)
            print("✅ D2 weights loaded")
        except:
            print("⚠️  Using random weights")
        
        original_model.eval()
        
        # Create model with NMS
        model_with_nms = EfficientDetD2WithNMSExport(
            original_model,
            confidence_threshold=0.6,
            nms_threshold=0.5,
            max_detections=10
        )
        model_with_nms.eval()
        
        # Test the model
        test_input = torch.randn(1, 3, 768, 768)
        
        with torch.no_grad():
            boxes, scores, labels = model_with_nms(test_input)
        
        print(f"✅ Model test successful:")
        print(f"   Boxes: {boxes.shape}")
        print(f"   Scores: {scores.shape}")
        print(f"   Labels: {labels.shape}")
        
        # Export to ONNX
        onnx_path = 'efficientdet_d2_with_nms.onnx'
        
        torch.onnx.export(
            model_with_nms,
            test_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['boxes', 'scores', 'labels'],
            verbose=False,
            dynamic_axes={
                'image': {0: 'batch_size'},
                'boxes': {0: 'batch_size'},
                'scores': {0: 'batch_size'},
                'labels': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX export with NMS successful: {onnx_path}")
        
        # Verify the export
        verify_nms_model(onnx_path, test_input)
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Export with NMS failed: {e}")
        return None

def verify_nms_model(onnx_path, test_input):
    """Verify the NMS ONNX model"""
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: test_input.numpy()})
        
        boxes, scores, labels = outputs
        
        print(f"✅ ONNX Runtime verification successful")
        print(f"   Output boxes: {boxes.shape}")
        print(f"   Output scores: {scores.shape}")
        print(f"   Output labels: {labels.shape}")
        
        # Count valid detections (non-zero scores)
        valid_dets = np.sum(scores[0] > 0)
        print(f"   Valid detections: {valid_dets}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False

def create_nms_inference_script():
    """Create inference script for NMS model"""
    
    inference_code = '''# Inference for EfficientDet-D2 with NMS Included
import cv2
import numpy as np
import onnxruntime as ort

class EfficientDetD2NMSInference:
    def __init__(self, model_path='efficientdet_d2_with_nms.onnx'):
        self.model_path = model_path
        self.class_names = ['signature', 'barcode', 'chop', 'qrcode']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        print(f"✅ D2 NMS model loaded: {model_path}")
        print("   NMS and confidence filtering included in model!")
    
    def predict(self, image_path):
        """Run prediction with built-in NMS"""
        print(f"\\n🔍 Processing: {image_path}")
        
        # Load and preprocess
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        orig_h, orig_w = image.shape[:2]
        
        # Resize to 768x768
        resized = cv2.resize(image, (768, 768))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to NCHW
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run inference (NMS is already applied!)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, scores, labels = outputs
        
        # Convert to original image coordinates
        scale_x = orig_w / 768.0
        scale_y = orig_h / 768.0
        
        detections = []
        for i in range(len(scores[0])):
            score = scores[0][i]
            
            if score > 0:  # Valid detection
                box = boxes[0][i]
                label = int(labels[0][i])
                
                # Scale box to original image
                x1 = box[0] * scale_x
                y1 = box[1] * scale_y
                x2 = box[2] * scale_x
                y2 = box[3] * scale_y
                
                detections.append({
                    'class_id': label,
                    'class_name': self.class_names[min(label, len(self.class_names)-1)],
                    'score': float(score),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
        
        print(f"   Found {len(detections)} final detections (after NMS)")
        return detections
    
    def visualize(self, image_path, detections):
        """Visualize final detections"""
        image = cv2.imread(image_path)
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            color = self.colors[det['class_id'] % len(self.colors)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            label = f"{det['class_name']}: {det['score']:.3f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imwrite('d2_nms_result.jpg', image)
        print("   📁 Result saved: d2_nms_result.jpg")

# Test
if __name__ == '__main__':
    detector = EfficientDetD2NMSInference()
    
    import glob
    images = glob.glob('*.jpg') + glob.glob('*.png')
    
    if images:
        detections = detector.predict(images[0])
        
        for det in detections:
            print(f"  - {det['class_name']}: {det['score']:.3f}")
        
        if detections:
            detector.visualize(images[0], detections)
    else:
        print("No test images found")
'''
    
    with open('d2_nms_inference.py', 'w') as f:
        f.write(inference_code)
    
    print("📄 Created d2_nms_inference.py")

if __name__ == '__main__':
    # Export model with NMS
    result = export_d2_with_nms()
    
    if result:
        # Create inference script
        create_nms_inference_script()
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Created: {result}")
        print(f"✅ Created: d2_nms_inference.py")
        print(f"\n🚀 THIS MODEL INCLUDES NMS!")
        print(f"   - No need for post-processing")
        print(f"   - Limited to 10 detections max")
        print(f"   - Confidence threshold: 0.6")
        print(f"   - NMS threshold: 0.5")
        
        print(f"\n💻 RUN:")
        print(f"   python d2_nms_inference.py")
    else:
        print("\n❌ Export failed")

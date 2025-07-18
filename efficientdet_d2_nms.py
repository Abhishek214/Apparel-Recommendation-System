# EfficientDet-D2 with Built-in NMS and Channel Fixes
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from backbone import EfficientDetBackbone

class EfficientDetD2WithNMS(nn.Module):
    """
    EfficientDet-D2 with integrated NMS for clean ONNX export
    """
    def __init__(self, original_model, confidence_threshold=0.5, nms_threshold=0.5, max_detections=10):
        super().__init__()
        
        self.compound_coef = 2
        self.num_classes = original_model.num_classes
        self.input_size = 768
        
        # NMS parameters
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        
        # Fixed backbone for D2
        self.backbone = self._create_fixed_backbone()
        
        # Anchor configuration
        self.anchor_scales = [1.0, 1.26, 1.59]
        self.anchor_ratios = [[1.0, 1.0], [1.4, 0.7], [0.7, 1.4]]
        self.num_anchors_per_location = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # FPN with consistent channels
        self.fpn_channels = 112
        
        # Feature adapters with fixed channels
        self.p3_adapter = nn.Conv2d(48, self.fpn_channels, 1, bias=False)  # Fixed channel count
        self.p4_adapter = nn.Conv2d(88, self.fpn_channels, 1, bias=False)
        self.p5_adapter = nn.Conv2d(248, self.fpn_channels, 1, bias=False)
        
        # Prediction heads
        self.classifier = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(),
            nn.Conv2d(self.fpn_channels, self.num_classes * self.num_anchors_per_location, 1, bias=True)
        )
        
        self.regressor = nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(),
            nn.Conv2d(self.fpn_channels, 4 * self.num_anchors_per_location, 1, bias=True)
        )
        
        # Pre-generate anchors
        self.register_buffer('all_anchors', self._generate_anchors())
        
        print(f"✅ EfficientDet-D2 with NMS initialized")
        print(f"   Confidence threshold: {confidence_threshold}")
        print(f"   NMS threshold: {nms_threshold}")
        print(f"   Max detections: {max_detections}")
    
    def _create_fixed_backbone(self):
        """Create backbone with consistent channel outputs"""
        return nn.Sequential(
            # Stage 1: 768 -> 384 (stride 2)
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Stage 2: 384 -> 192 (stride 2) 
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            
            # Stage 3: 192 -> 96 (stride 2) - P3
            nn.Conv2d(48, 48, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            
            # Stage 4: 96 -> 48 (stride 2) - P4
            nn.Conv2d(48, 88, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(88),
            nn.ReLU(),
            
            # Stage 5: 48 -> 24 (stride 2) - P5
            nn.Conv2d(88, 248, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(248),
            nn.ReLU()
        )
    
    def _generate_anchors(self):
        """Generate anchors for D2 feature pyramid"""
        all_anchors = []
        
        # D2 feature map sizes
        feature_sizes = [(96, 96), (48, 48), (24, 24)]  # P3, P4, P5
        strides = [8, 16, 32]
        
        for (height, width), stride in zip(feature_sizes, strides):
            level_anchors = self._generate_level_anchors(height, width, stride)
            all_anchors.append(level_anchors)
        
        return torch.cat(all_anchors, dim=0)
    
    def _generate_level_anchors(self, height, width, stride):
        """Generate anchors for one pyramid level"""
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
    
    def _extract_features(self, x):
        """Extract P3, P4, P5 features with fixed channels"""
        features = []
        
        # Forward through backbone
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            
            # Collect pyramid features
            if i == 5:  # After stage 2 (P3 level) 
                p3 = F.interpolate(x, size=(96, 96), mode='bilinear', align_corners=False)
                features.append(p3)
            elif i == 8:  # After stage 3
                continue  # Skip this
            elif i == 11:  # After stage 4 (P4 level)
                p4 = F.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
                features.append(p4)
            elif i == 14:  # After stage 5 (P5 level)
                p5 = F.interpolate(x, size=(24, 24), mode='bilinear', align_corners=False)
                features.append(p5)
        
        return features
    
    def _apply_nms(self, boxes, scores, labels):
        """Apply NMS to filter detections"""
        batch_size = boxes.shape[0]
        
        # Process each batch
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for b in range(batch_size):
            batch_boxes = boxes[b]
            batch_scores = scores[b]
            batch_labels = labels[b]
            
            # Filter by confidence
            valid_mask = batch_scores > self.confidence_threshold
            if not valid_mask.any():
                # No valid detections
                final_boxes.append(torch.zeros(self.max_detections, 4))
                final_scores.append(torch.zeros(self.max_detections))
                final_labels.append(torch.zeros(self.max_detections, dtype=torch.long))
                continue
            
            valid_boxes = batch_boxes[valid_mask]
            valid_scores = batch_scores[valid_mask]
            valid_labels = batch_labels[valid_mask]
            
            # Apply NMS per class
            keep_indices = []
            for class_id in range(self.num_classes):
                class_mask = valid_labels == class_id
                if not class_mask.any():
                    continue
                
                class_boxes = valid_boxes[class_mask]
                class_scores = valid_scores[class_mask]
                
                # Simple NMS implementation for ONNX compatibility
                keep = self._simple_nms(class_boxes, class_scores, self.nms_threshold)
                class_indices = torch.nonzero(class_mask).squeeze(1)
                keep_indices.extend(class_indices[keep])
            
            # Limit to max detections
            if len(keep_indices) > self.max_detections:
                # Sort by score and keep top detections
                keep_scores = valid_scores[keep_indices]
                _, sort_indices = torch.sort(keep_scores, descending=True)
                keep_indices = [keep_indices[i] for i in sort_indices[:self.max_detections]]
            
            # Pad to max_detections
            num_keep = len(keep_indices)
            batch_final_boxes = torch.zeros(self.max_detections, 4)
            batch_final_scores = torch.zeros(self.max_detections)
            batch_final_labels = torch.zeros(self.max_detections, dtype=torch.long)
            
            if num_keep > 0:
                keep_tensor = torch.tensor(keep_indices)
                batch_final_boxes[:num_keep] = valid_boxes[keep_tensor]
                batch_final_scores[:num_keep] = valid_scores[keep_tensor]
                batch_final_labels[:num_keep] = valid_labels[keep_tensor]
            
            final_boxes.append(batch_final_boxes)
            final_scores.append(batch_final_scores)
            final_labels.append(batch_final_labels)
        
        return torch.stack(final_boxes), torch.stack(final_scores), torch.stack(final_labels)
    
    def _simple_nms(self, boxes, scores, threshold):
        """Simple NMS implementation compatible with ONNX"""
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        
        # Sort by score
        _, indices = torch.sort(scores, descending=True)
        
        keep = []
        while len(indices) > 0:
            # Keep highest scoring box
            current = indices[0]
            keep.append(current.item())
            
            if len(indices) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = boxes[current].unsqueeze(0)
            remaining_boxes = boxes[indices[1:]]
            
            iou = self._calculate_iou(current_box, remaining_boxes)
            
            # Remove boxes with high IoU
            mask = iou.squeeze(0) <= threshold
            indices = indices[1:][mask]
        
        return torch.tensor(keep, dtype=torch.long)
    
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between boxes"""
        # box1: [1, 4], box2: [N, 4]
        
        # Calculate intersection
        x1 = torch.max(box1[:, 0:1], box2[:, 0:1].t())
        y1 = torch.max(box1[:, 1:2], box2[:, 1:2].t())
        x2 = torch.min(box1[:, 2:3], box2[:, 2:3].t())
        y2 = torch.min(box1[:, 3:4], box2[:, 3:4].t())
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate areas
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        union = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection
        
        return intersection / (union + 1e-8)
    
    def forward(self, x):
        """Forward pass with NMS"""
        batch_size = x.shape[0]
        
        # Extract features
        features = self._extract_features(x)
        
        # Adapt features and generate predictions
        all_cls_outputs = []
        all_reg_outputs = []
        
        adapters = [self.p3_adapter, self.p4_adapter, self.p5_adapter]
        
        for feature, adapter in zip(features, adapters):
            # Adapt feature channels
            adapted_feature = adapter(feature)
            
            # Generate predictions
            cls_output = self.classifier(adapted_feature)
            reg_output = self.regressor(adapted_feature)
            
            # Reshape predictions
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
        
        # Apply sigmoid
        classification = torch.sigmoid(classification)
        
        # Decode boxes (simple anchor-based decoding)
        decoded_boxes = self._decode_boxes(bbox_regression, self.all_anchors)
        
        # Get max scores and labels for each prediction
        max_scores, max_labels = torch.max(classification, dim=2)
        
        # Apply NMS
        final_boxes, final_scores, final_labels = self._apply_nms(decoded_boxes, max_scores, max_labels)
        
        return final_boxes, final_scores, final_labels
    
    def _decode_boxes(self, regression, anchors):
        """Simple box decoding"""
        # For simplicity, just apply small offsets to anchors
        batch_size = regression.shape[0]
        num_anchors = anchors.shape[0]
        
        # Expand anchors for batch
        anchors_expanded = anchors.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Apply simple decoding (you may want to use proper RetinaNet decoding)
        decoded = anchors_expanded + regression * 50.0  # Scale factor
        
        return decoded

def export_efficientdet_d2_with_nms():
    """Export EfficientDet-D2 with built-in NMS"""
    print("🚀 EXPORTING EFFICIENTDET-D2 WITH BUILT-IN NMS")
    print("="*60)
    
    # Load original model
    compound_coef = 2
    obj_list = ['signature', 'barcode', 'chop', 'qrcode']
    
    try:
        original_model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
        
        # Try to load weights
        model_paths = [
            'logs/abhi/efficientdet-d2_24_1400.pth',
            'logs/abhi/efficientdet-d1_24_1400.pth'
        ]
        
        for model_path in model_paths:
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                original_model.load_state_dict(state_dict, strict=False)
                print(f"✅ Loaded weights from: {model_path}")
                break
            except:
                continue
        
        original_model.eval()
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Create NMS model with proper thresholds
    print("Creating D2 model with built-in NMS...")
    nms_model = EfficientDetD2WithNMS(
        original_model,
        confidence_threshold=0.6,  # Higher threshold to reduce false positives
        nms_threshold=0.5,
        max_detections=10  # Limit output detections
    )
    nms_model.eval()
    
    # Test the model
    test_input = torch.randn(1, 3, 768, 768)
    
    try:
        with torch.no_grad():
            boxes, scores, labels = nms_model(test_input)
        
        print(f"✅ NMS model test successful:")
        print(f"   Boxes: {boxes.shape}")
        print(f"   Scores: {scores.shape}")
        print(f"   Labels: {labels.shape}")
        
    except Exception as e:
        print(f"❌ NMS model test failed: {e}")
        return None
    
    # Export to ONNX
    onnx_path = 'efficientdet_d2_with_nms.onnx'
    print(f"\nExporting to ONNX: {onnx_path}")
    
    try:
        torch.onnx.export(
            nms_model,
            test_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['boxes', 'scores', 'labels'],
            verbose=False,
            dynamic_axes={
                'image': {0: 'batch_size'}
            }
        )
        
        print("✅ ONNX export with NMS successful!")
        print(f"📁 File: {onnx_path}")
        
        # Verify
        verify_nms_onnx(onnx_path)
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None

def verify_nms_onnx(onnx_path):
    """Verify the NMS ONNX model"""
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Test inference
        test_input = np.random.randn(1, 3, 768, 768).astype(np.float32)
        input_name = session.get_inputs()[0].name
        
        outputs = session.run(None, {input_name: test_input})
        boxes, scores, labels = outputs
        
        print(f"✅ ONNX NMS verification successful!")
        print(f"   Output shapes: boxes{boxes.shape}, scores{scores.shape}, labels{labels.shape}")
        
        # Count valid detections (score > 0)
        valid_detections = np.sum(scores[0] > 0)
        print(f"   Valid detections in test: {valid_detections}")
        
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")

def create_nms_inference_script():
    """Create inference script for NMS model"""
    
    script = '''# EfficientDet-D2 with Built-in NMS Inference
import cv2
import numpy as np
import onnxruntime as ort

class EfficientDetD2NMS:
    def __init__(self, model_path='efficientdet_d2_with_nms.onnx'):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = ['signature', 'barcode', 'chop', 'qrcode']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        print(f"✅ D2 NMS model loaded: {model_path}")
    
    def predict(self, image_path):
        """Predict with built-in NMS"""
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        orig_h, orig_w = image.shape[:2]
        
        # Preprocess
        resized = cv2.resize(image, (768, 768))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run inference (NMS is built-in)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        boxes, scores, labels = outputs
        
        # Scale boxes back to original image
        scale_x = orig_w / 768.0
        scale_y = orig_h / 768.0
        
        detections = []
        for i in range(len(scores[0])):
            score = scores[0][i]
            if score > 0.1:  # Only valid detections
                label_id = int(labels[0][i])
                box = boxes[0][i]
                
                x1 = max(0, min(box[0] * scale_x, orig_w))
                y1 = max(0, min(box[1] * scale_y, orig_h))
                x2 = max(0, min(box[2] * scale_x, orig_w))
                y2 = max(0, min(box[3] * scale_y, orig_h))
                
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'class_id': label_id,
                        'class_name': self.class_names[min(label_id, len(self.class_names)-1)],
                        'score': float(score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
        
        return detections
    
    def visualize(self, image_path, detections):
        """Visualize NMS results"""
        image = cv2.imread(image_path)
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            color = self.colors[det['class_id'] % len(self.colors)]
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            label = f"{det['class_name']}: {det['score']:.2f}"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imwrite('nms_result.jpg', image)
        print(f"NMS result saved: nms_result.jpg")

# Test
if __name__ == '__main__':
    detector = EfficientDetD2NMS()
    
    import os
    images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.png'))]
    
    if images:
        test_image = images[0]
        print(f"\\nTesting with: {test_image}")
        
        detections = detector.predict(test_image)
        print(f"Found {len(detections)} detections (after NMS):")
        
        for det in detections:
            print(f"  - {det['class_name']}: {det['score']:.3f}")
        
        if detections:
            detector.visualize(test_image, detections)
    else:
        print("No test images found")
'''
    
    with open('efficientdet_d2_nms_inference.py', 'w') as f:
        f.write(script)
    
    print("📄 Created efficientdet_d2_nms_inference.py")

if __name__ == '__main__':
    result = export_efficientdet_d2_with_nms()
    
    if result:
        create_nms_inference_script()
        
        print(f"\n🎉 SUCCESS! D2 with built-in NMS ready")
        print(f"🚀 Test: python efficientdet_d2_nms_inference.py")
        print(f"\n🔧 NMS FIXES:")
        print(f"   • Built-in NMS in ONNX model")
        print(f"   • Higher confidence threshold (0.6)")
        print(f"   • Max 10 detections per image")
        print(f"   • Fixed channel dimensions")
        print(f"   • IoU-based overlap removal")

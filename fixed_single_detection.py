import cv2
import numpy as np
import onnxruntime as ort

def _apply_nms_single_box(self, boxes, scores, score_threshold=0.3, nms_threshold=0.3):
    """
    Improved NMS that ensures only one box per object
    """
    # Filter by score threshold first
    valid_indices = scores > score_threshold
    if not np.any(valid_indices):
        return [], [], []
    
    filtered_boxes = boxes[valid_indices]
    filtered_scores = scores[valid_indices]
    
    # Sort by scores (highest first)
    order = filtered_scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if order.size == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = filtered_boxes[i]
        remaining_boxes = filtered_boxes[order[1:]]
        
        # IoU calculation
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        # Intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Union area
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - intersection
        
        # IoU
        iou = intersection / (union + 1e-6)
        
        # Keep boxes with IoU less than threshold
        indices = np.where(iou <= nms_threshold)[0]
        order = order[indices + 1]
    
    # Return only the highest scoring box (single detection)
    if len(keep) > 0:
        best_idx = 0  # Already sorted by score
        return [filtered_boxes[keep[best_idx]]], [filtered_scores[keep[best_idx]]], [keep[best_idx]]
    else:
        return [], [], []

def _postprocess_detections_single(self, regression, classification, metadata, 
                                 sig_thresh=0.5, thresh=0.3):
    """
    Post-process to ensure single detection per class
    """
    # Apply sigmoid to classification scores
    scores = 1 / (1 + np.exp(-classification))
    
    # Apply regression to anchors (simplified)
    regression = regression[0]  # Remove batch dimension
    scores = scores[0]  # Remove batch dimension
    
    # Convert regression to boxes (same as before)
    boxes = self.anchors.copy()
    
    # Apply regression deltas
    dx = regression[:, 0]
    dy = regression[:, 1]
    dw = regression[:, 2]
    dh = regression[:, 3]
    
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    
    cx_new = cx + dx * w
    cy_new = cy + dy * h
    w_new = w * np.exp(np.clip(dw, -5, 5))  # Clip to avoid extreme values
    h_new = h * np.exp(np.clip(dh, -5, 5))
    
    x1 = cx_new - w_new / 2
    y1 = cy_new - h_new / 2
    x2 = cx_new + w_new / 2
    y2 = cy_new + h_new / 2
    
    final_boxes = np.stack([x1, y1, x2, y2], axis=1)
    final_boxes *= self.input_size
    
    detections = []
    
    # Process each class and keep only the best detection
    for class_idx, class_name in enumerate(self.classes):
        class_scores = scores[:, class_idx]
        
        # Use stricter thresholds
        if class_name == 'signature':
            current_thresh = max(sig_thresh, 0.5)  # Minimum 0.5 for signatures
        else:
            current_thresh = max(thresh, 0.4)      # Minimum 0.4 for others
        
        # Apply NMS to get single best detection
        nms_boxes, nms_scores, nms_indices = self._apply_nms_single_box(
            final_boxes, class_scores, 
            score_threshold=current_thresh, 
            nms_threshold=0.3  # Stricter NMS
        )
        
        # Should only have one box per class now
        for box, score in zip(nms_boxes, nms_scores):
            # Convert coordinates back to original image space
            x1 = (box[0] - metadata['pad_x']) / metadata['scale']
            y1 = (box[1] - metadata['pad_y']) / metadata['scale']
            x2 = (box[2] - metadata['pad_x']) / metadata['scale']
            y2 = (box[3] - metadata['pad_y']) / metadata['scale']
            
            # Clip to image bounds
            x1 = max(0, min(x1, metadata['original_width']))
            y1 = max(0, min(y1, metadata['original_height']))
            x2 = max(0, min(x2, metadata['original_width']))
            y2 = max(0, min(y2, metadata['original_height']))
            
            # Ensure valid box
            if x2 > x1 and y2 > y1:
                detections.append({
                    'class': class_name,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score)
                })
    
    return detections

# Modified DetectImage function with single detection guarantee
def DetectImage(self, image, sig_thresh=0.5, thresh=0.3, short=512):
    """
    Modified to ensure single detection per class
    """
    try:
        # Preprocess image
        img_input, metadata = self._preprocess_image(image)
        
        # Run ONNX inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_input})
        
        regression = outputs[0]
        classification = outputs[1]
        
        # Post-process with single detection per class
        detections = self._postprocess_detections_single(
            regression, classification, metadata, sig_thresh, thresh
        )
        
        # Convert to your existing format
        Items = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Additional validation - ensure reasonable box size
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height
            
            # Filter tiny boxes (likely false positives)
            if box_area > 100:  # Minimum area threshold
                Items.append({
                    'class': detection['class'],
                    'bbox': [y1, x2, x1, x2],  # Your format
                    'score': str(round(detection['score'] * 100, 2))
                })
        
        # Additional filtering: Keep only the highest scoring detection per class
        class_detections = {}
        for item in Items:
            class_name = item['class']
            score = float(item['score'])
            
            if class_name not in class_detections or score > float(class_detections[class_name]['score']):
                class_detections[class_name] = item
        
        # Final result: maximum one detection per class
        final_items = list(class_detections.values())
        
        self.logger.info(f"Final single detections: {final_items}")
        
        # Save image and return results
        if isinstance(image, np.ndarray):
            pil_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            pil_image = cv2.imread(image)
        
        cv2.imwrite("./CurrentPAss.png", pil_image)
        
        return final_items
        
    except Exception as e:
        self.logger.error(f"Error in DetectImage: {e}")
        return []

# Usage with stricter thresholds for single detection
def get_single_detection_results(detector, image_path):
    """
    Helper function to get single detection results
    """
    results = detector.DetectImage(
        image_path, 
        sig_thresh=0.6,  # Higher threshold for signatures
        thresh=0.5       # Higher threshold for others
    )
    
    print(f"Single detection results: {len(results)} objects found")
    for result in results:
        print(f"  - {result['class']}: {result['score']}% confidence")
    
    return results

import cv2
import numpy as np
import onnxruntime as ort
import logging

class EfficientDetONNXDetector:
    def __init__(self, onnx_model_path):
        """
        Initialize EfficientDet D2 ONNX detector
        """
        self.onnx_model_path = onnx_model_path
        self.input_size = 768  # D2 input size
        self.classes = ['signature', 'barcode', 'chop', 'qrcode']  # Your 4 classes
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Load ONNX model
        self._load_model()
        
        # Generate anchors (since they're not in ONNX export)
        self.anchors = self._generate_anchors()
    
    def _load_model(self):
        """Load ONNX model"""
        try:
            self.session = ort.InferenceSession(self.onnx_model_path)
            self.logger.info(f"EfficientDet D2 ONNX model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _generate_anchors(self):
        """Generate anchors for EfficientDet D2"""
        # D2 configuration
        anchor_scale = 4.0
        aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        feature_sizes = [96, 48, 24, 12, 6]  # Feature map sizes for 768x768 input
        
        anchors = []
        for level, size in enumerate(feature_sizes):
            level_anchors = self._generate_level_anchors(
                size, size, anchor_scale * (2 ** (level / 3.0)), 
                aspect_ratios, anchor_scales
            )
            anchors.append(level_anchors)
        
        return np.concatenate(anchors, axis=0)
    
    def _generate_level_anchors(self, height, width, scale, ratios, scales):
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
    
    def _preprocess_image(self, image):
        """
        Preprocess image for EfficientDet D2
        Returns preprocessed image and metadata for coordinate conversion
        """
        if isinstance(image, str):
            # Image path
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Numpy array
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR format from OpenCV
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                img = image
        
        original_height, original_width = img.shape[:2]
        
        # Calculate scale to maintain aspect ratio
        scale = min(self.input_size / original_width, self.input_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded_img = np.ones((self.input_size, self.input_size, 3), dtype=np.uint8) * 114
        
        # Center the image
        pad_x = (self.input_size - new_width) // 2
        pad_y = (self.input_size - new_height) // 2
        padded_img[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = img_resized
        
        # Normalize using ImageNet statistics
        img_normalized = padded_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std
        
        # Convert to CHW format and add batch dimension
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_input, axis=0)
        
        # Metadata for coordinate conversion
        metadata = {
            'original_width': original_width,
            'original_height': original_height,
            'scale': scale,
            'pad_x': pad_x,
            'pad_y': pad_y
        }
        
        return img_batch.astype(np.float32), metadata
    
    def _apply_nms(self, boxes, scores, score_threshold=0.3, nms_threshold=0.5):
        """Apply Non-Maximum Suppression"""
        # Filter by score threshold
        valid_indices = scores > score_threshold
        if not np.any(valid_indices):
            return [], [], []
        
        filtered_boxes = boxes[valid_indices]
        filtered_scores = scores[valid_indices]
        valid_indices_list = np.where(valid_indices)[0]
        
        # Convert to x1, y1, x2, y2 format for NMS
        x1 = filtered_boxes[:, 0]
        y1 = filtered_boxes[:, 1]
        x2 = filtered_boxes[:, 2]
        y2 = filtered_boxes[:, 3]
        
        # Calculate areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by scores
        order = filtered_scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Keep boxes with IoU less than threshold
            indices = np.where(iou <= nms_threshold)[0]
            order = order[indices + 1]
        
        return (filtered_boxes[keep], 
                filtered_scores[keep], 
                valid_indices_list[keep])
    
    def _postprocess_detections(self, regression, classification, metadata, 
                              sig_thresh=0.5, thresh=0.3):
        """
        Post-process EfficientDet outputs to get final detections
        """
        # Apply sigmoid to classification scores
        scores = 1 / (1 + np.exp(-classification))  # Sigmoid
        
        # Apply regression to anchors to get final boxes
        # Regression format: [dx, dy, dw, dh]
        regression = regression[0]  # Remove batch dimension
        scores = scores[0]  # Remove batch dimension
        
        # Convert regression deltas to actual boxes
        boxes = self.anchors.copy()
        
        # Apply regression deltas (simplified - you may need more complex transformation)
        dx = regression[:, 0]
        dy = regression[:, 1]
        dw = regression[:, 2]
        dh = regression[:, 3]
        
        # Convert from center format to corner format
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        
        # Apply deltas
        cx_new = cx + dx * w
        cy_new = cy + dy * h
        w_new = w * np.exp(dw)
        h_new = h * np.exp(dh)
        
        # Convert back to corner format
        x1 = cx_new - w_new / 2
        y1 = cy_new - h_new / 2
        x2 = cx_new + w_new / 2
        y2 = cy_new + h_new / 2
        
        final_boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        # Convert to pixel coordinates
        final_boxes *= self.input_size
        
        detections = []
        
        # Process each class
        for class_idx, class_name in enumerate(self.classes):
            class_scores = scores[:, class_idx]
            
            # Apply NMS for this class
            if class_name == 'signature':
                current_thresh = sig_thresh
            else:
                current_thresh = thresh
            
            nms_boxes, nms_scores, nms_indices = self._apply_nms(
                final_boxes, class_scores, 
                score_threshold=current_thresh, 
                nms_threshold=0.5
            )
            
            # Convert coordinates back to original image space
            for box, score in zip(nms_boxes, nms_scores):
                # Remove padding and scale back
                x1 = (box[0] - metadata['pad_x']) / metadata['scale']
                y1 = (box[1] - metadata['pad_y']) / metadata['scale']
                x2 = (box[2] - metadata['pad_x']) / metadata['scale']
                y2 = (box[3] - metadata['pad_y']) / metadata['scale']
                
                # Clip to image bounds
                x1 = max(0, min(x1, metadata['original_width']))
                y1 = max(0, min(y1, metadata['original_height']))
                x2 = max(0, min(x2, metadata['original_width']))
                y2 = max(0, min(y2, metadata['original_height']))
                
                detections.append({
                    'class': class_name,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score)
                })
        
        return detections

# Modified DetectImage function
def DetectImage(self, image, sig_thresh=0.5, thresh=0.3, short=512):
    """
    Modified DetectImage function for EfficientDet D2 ONNX model
    
    Args:
        image: Input image (numpy array or image path)
        sig_thresh: Threshold for signature detection (default 0.5)
        thresh: Threshold for other objects (default 0.3) 
        short: Not used in ONNX version (kept for compatibility)
    
    Returns:
        Detection results in the same format as before
    """
    try:
        # Preprocess image
        img_input, metadata = self._preprocess_image(image)
        
        # Run ONNX inference
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img_input})
        
        # Extract regression and classification outputs
        regression = outputs[0]  # Shape: [1, num_anchors, 4]
        classification = outputs[1]  # Shape: [1, num_anchors, num_classes]
        
        # Post-process detections
        detections = self._postprocess_detections(
            regression, classification, metadata, sig_thresh, thresh
        )
        
        # Convert to your existing format
        Items = []
        for detection in detections:
            if detection['class'].lower() == 'signature':
                # Filter signatures that meet threshold
                if detection['score'] >= sig_thresh:
                    x1, y1, x2, y2 = detection['bbox']
                    Items.append({
                        'class': detection['class'],
                        'bbox': [y1, x2, x1, x2],  # Match your format: [ymin, xmax, xmin, xmax]
                        'score': str(round(detection['score'] * 100, 2))
                    })
            else:
                # Other classes (barcode, chop, qrcode)
                if detection['score'] >= thresh:
                    x1, y1, x2, y2 = detection['bbox']
                    Items.append({
                        'class': detection['class'],
                        'bbox': [y1, x2, x1, x2],  # Match your format
                        'score': str(round(detection['score'] * 100, 2))
                    })
        
        # Log detections
        Items_filtered = [item for item in Items if item['class'].lower() == 'signature']
        self.logger.info(f"Detections of signature, chop and qr code data are {Items}")
        
        # Convert image for saving (if needed)
        if isinstance(image, np.ndarray):
            pil_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            pil_image = cv2.imread(image)
        
        # Save current detection result (matching your existing code)
        cv2.imwrite("./CurrentPAss.png", pil_image)
        
        # Prepare final detection data
        det_data = Items
        self.logger.info(f"DETECTION DATA IS {det_data}")
        
        # Convert image to bytes (if needed for API response)
        src_img_bytes = cv2.imencode('.png', pil_image)[1].tobytes()
        
        final_det_resp_data = det_data
        
        return final_det_resp_data
        
    except Exception as e:
        self.logger.error(f"Error in DetectImage: {e}")
        return []

# Usage example and initialization code
class DocumentDetector:
    """
    Main detector class that replaces your existing MXNet-based detector
    """
    def __init__(self, onnx_model_path="efficientdet_d2_stride_fixed.onnx"):
        """
        Initialize the document detector with EfficientDet D2 ONNX model
        """
        self.detector = EfficientDetONNXDetector(onnx_model_path)
        self.logger = logging.getLogger(__name__)
    
    def DetectImage(self, image, sig_thresh=0.5, thresh=0.3, short=512):
        """
        Public interface that matches your existing function signature
        """
        return self.detector.DetectImage(image, sig_thresh, thresh, short)

# How to replace your existing detector:
# 1. Replace your MXNet model initialization with:
# detector = DocumentDetector("path/to/your/efficientdet_d2_stride_fixed.onnx")

# 2. Use the same function call:
# results = detector.DetectImage(image, sig_thresh=0.5, thresh=0.3)

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = DocumentDetector("efficientdet_d2_stride_fixed.onnx")
    
    # Test detection
    test_image = "test_document.jpg"
    results = detector.DetectImage(test_image, sig_thresh=0.5, thresh=0.3)
    
    print(f"Detection results: {results}")

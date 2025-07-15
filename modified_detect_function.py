def DetectImage(self, image, sig_thresh=0.6, thresh=0.4, short=512):
    """
    Detect table on input image or image path
    :param image: numpy array or image path
    :param thresh: default 0.50, can be changed
    
    Modified to use EfficientDet ONNX model instead of MXNet
    """
    
    # Configuration for EfficientDet
    input_size = 640  # For D1 model (use 768 for D2)
    
    # Convert image path to numpy array if needed
    if isinstance(image, str):
        image = cv2.imread(image)
    
    original_height, original_width = image.shape[:2]
    
    # EfficientDet preprocessing
    # 1. Resize to model input size while maintaining aspect ratio
    scale = min(input_size / original_width, input_size / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    img_resized = cv2.resize(image, (new_width, new_height))
    
    # 2. Pad to square
    img_padded = np.ones((input_size, input_size, 3), dtype=np.uint8) * 114
    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2
    img_padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = img_resized
    
    # 3. Convert to RGB and normalize
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # 4. Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    
    # 5. Convert to NCHW format for ONNX
    img_input = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_input, axis=0)
    
    # Run ONNX inference
    outputs = self.onnx_session.run(None, {self.input_name: img_batch})
    regression, classification = outputs
    
    # Generate anchors (since ONNX model doesn't include them)
    anchors = self._generate_anchors_for_inference(input_size)
    
    # Apply sigmoid to classification scores
    classification_scores = 1 / (1 + np.exp(-classification))
    
    # Apply regression to anchors to get bounding boxes
    bounding_boxes = self._apply_regression_to_anchors(regression[0], anchors)
    
    # Get class predictions and scores
    max_scores = np.max(classification_scores[0], axis=1)
    class_IDs = np.argmax(classification_scores[0], axis=1)
    
    # Apply confidence threshold
    valid_indices = max_scores > thresh
    
    scores_list = max_scores[valid_indices]
    bbox_list = bounding_boxes[valid_indices]
    cls_id_list = class_IDs[valid_indices]
    
    # Apply NMS (Non-Maximum Suppression)
    if len(scores_list) > 0:
        keep_indices = self._apply_nms(bbox_list, scores_list, nms_thresh=0.5)
        scores_list = scores_list[keep_indices]
        bbox_list = bbox_list[keep_indices]
        cls_id_list = cls_id_list[keep_indices]
    
    # Convert coordinates back to original image space
    Items = []
    for score, bbox, classes in zip(scores_list, bbox_list, cls_id_list):
        if score > thresh:
            # Convert from padded coordinates to original coordinates
            x1, y1, x2, y2 = bbox
            
            # Remove padding offset
            x1 = max(0, (x1 - pad_x) / scale)
            y1 = max(0, (y1 - pad_y) / scale)
            x2 = min(original_width, (x2 - pad_x) / scale)
            y2 = min(original_height, (y2 - pad_y) / scale)
            
            # Skip invalid detections
            if self.classes[classes].lower() == 'signature':
                continue
            else:
                xmin, ymin, xmax, ymax = self.convert_bbox_to_original_size(
                    [x1, y1, x2, y2], image.shape, image.shape
                )
                Items.append({
                    "class": self.classes[classes],
                    "bbox": [ymin, ymax, xmin, xmax],
                    "score": str(round(score * 100, 2))
                })
    
    # Process signature detections separately with different threshold
    sig_scores_list, sig_bbox_list, sig_cls_id_list = [], [], []
    
    for score, bbox, classes in zip(scores_list, bbox_list, cls_id_list):
        if score > sig_thresh:
            if self.classes[classes].lower() == 'signature':
                x1, y1, x2, y2 = bbox
                
                # Remove padding offset
                x1 = max(0, (x1 - pad_x) / scale)
                y1 = max(0, (y1 - pad_y) / scale)
                x2 = min(original_width, (x2 - pad_x) / scale)
                y2 = min(original_height, (y2 - pad_y) / scale)
                
                xmin, ymin, xmax, ymax = self.convert_bbox_to_original_size(
                    [x1, y1, x2, y2], image.shape, image.shape
                )
                Items.append({
                    "class": self.classes[classes],
                    "bbox": [ymin, ymax, xmin, xmax],
                    "score": str(round(score * 100, 2))
                })
    
    # Log detection information
    Items_filtered = [item for item in Items if item['class'].lower() != 'signature']
    self.logger.info(f"Detections of barcode, chop and qr code data are {Items_filtered}")
    
    Items_sig = [item for item in Items if item['class'].lower() == 'signature']
    self.logger.info(f"Detection data is {Items_sig}")
    
    # Convert image to bytes for return
    src_img_bytes = self.convert_image_to_file_bytes(image)
    det_data = Items
    
    final_det_rec_data = det_data
    
    return final_det_rec_data

def _generate_anchors_for_inference(self, input_size):
    """
    Generate anchors for EfficientDet inference
    """
    # Anchor configuration for your model
    anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    pyramid_levels = [3, 4, 5, 6, 7]  # Feature levels for D1
    anchor_scale = 4.0  # Base anchor scale
    
    anchors = []
    
    for level in pyramid_levels:
        feature_size = input_size // (2 ** level)
        
        for y in range(feature_size):
            for x in range(feature_size):
                cx = (x + 0.5) / feature_size
                cy = (y + 0.5) / feature_size
                
                for ratio in anchor_ratios:
                    for scale in anchor_scales:
                        w = anchor_scale * scale * ratio[0] / input_size
                        h = anchor_scale * scale * ratio[1] / input_size
                        
                        anchors.append([
                            cx - w/2, cy - h/2, cx + w/2, cy + h/2
                        ])
    
    return np.array(anchors, dtype=np.float32)

def _apply_regression_to_anchors(self, regression, anchors):
    """
    Apply regression predictions to anchors to get final bounding boxes
    """
    # Regression format: [dx, dy, dw, dh]
    dx = regression[:, 0]
    dy = regression[:, 1]
    dw = regression[:, 2]
    dh = regression[:, 3]
    
    # Anchor format: [x1, y1, x2, y2]
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
    
    # Apply regression
    pred_ctr_x = dx * anchor_widths + anchor_ctr_x
    pred_ctr_y = dy * anchor_heights + anchor_ctr_y
    pred_w = np.exp(dw) * anchor_widths
    pred_h = np.exp(dh) * anchor_heights
    
    # Convert back to [x1, y1, x2, y2] format
    pred_boxes = np.zeros_like(regression)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2
    
    # Convert from normalized coordinates to pixel coordinates
    input_size = 640  # Your model input size
    pred_boxes[:, [0, 2]] *= input_size  # x coordinates
    pred_boxes[:, [1, 3]] *= input_size  # y coordinates
    
    return pred_boxes

def _apply_nms(self, boxes, scores, nms_thresh=0.5):
    """
    Apply Non-Maximum Suppression
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Calculate areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # Sort by scores
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Calculate IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]
    
    return np.array(keep, dtype=np.int32)

# Additional initialization code to add to your class __init__ method:
def __init__(self):
    # Replace MXNet model loading with ONNX
    self.onnx_session = ort.InferenceSession('efficientdet_d1_stride_fixed.onnx')
    self.input_name = self.onnx_session.get_inputs()[0].name
    
    # Define your classes (adjust as needed)
    self.classes = ['signature', 'barcode', 'chop', 'qrcode']
    
    # Keep your existing logger and other initialization code
    # ...

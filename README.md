# Convert coordinates back to original image space
for box, score in zip(nms_boxes, nms_scores):
    # Remove padding and scale back
    x1 = (box[0] - metadata['pad_x']) / metadata['scale']
    y1 = (box[1] - metadata['pad_y']) / metadata['scale']
    x2 = (box[2] - metadata['pad_x']) / metadata['scale']
    y2 = (box[3] - metadata['pad_y']) / metadata['scale']
    
    # Ensure x1 < x2 and y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Clip to image bounds
    x1 = max(0, min(x1, metadata['original_width']))
    y1 = max(0, min(y1, metadata['original_height']))
    x2 = max(0, min(x2, metadata['original_width']))
    y2 = max(0, min(y2, metadata['original_height']))
    
    # Skip invalid boxes
    if x2 <= x1 or y2 <= y1:
        continue
    
    detections.append({
        'class': class_name,
        'bbox': [float(x1), float(y1), float(x2), float(y2)],
        'score': float(score)
    })🎯 Fix the Return Format:# In the DetectImage function, fix the bbox format:
Items.append({
    'class': detection['class'],
    'bbox': [y1, x2, x1, y2],  # Fixed: was [y1, x2, x1, x2]
    'score': str(round(detection['score'] * 100, 2))
})

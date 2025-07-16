    anchors = []
    
    for level in pyramid_levels:
        feature_size = input_size // (2 ** level)
        
        # Calculate stride for this level
        stride = 2 ** level
        
        for y in range(feature_size):
            for x in range(feature_size):
                # Center coordinates in original image space
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                
                for ratio_w, ratio_h in anchor_ratios:
                    for scale in anchor_scales:
                        # Calculate anchor size
                        anchor_size = anchor_scale * scale * stride
                        w = anchor_size * ratio_w
                        h = anchor_size * ratio_h
                        
                        # Convert to normalized coordinates [0, 1]
                        x1 = (cx - w/2) / input_size
                        y1 = (cy - h/2) / input_size
                        x2 = (cx + w/2) / input_size
                        y2 = (cy + h/2) / input_size
                        
                        anchors.append([x1, y1, x2, y2])
    
    anchors = np.array(anchors, dtype=np.float32)
    print(f"Generated {len(anchors)} anchors")  # Debug print
    return anchors

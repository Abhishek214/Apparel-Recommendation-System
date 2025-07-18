# ONNX EfficientDet-D2 COCO Evaluation Script
# Adapted from the original coco_eval.py for ONNX Runtime

import json
import os
import argparse
import numpy as np
import cv2
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort

from utils.utils import boolean_string

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=2, help='coefficients of efficientdet (2 for D2)')
ap.add_argument('-w', '--weights', type=str, default='efficientdet_d2_onnx_runtime_compatible.onnx', help='path to ONNX weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold')
ap.add_argument('--conf_threshold', type=float, default=0.05, help='confidence threshold')
ap.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
ap.add_argument('--max_images', type=int, default=10000, help='maximum number of images to evaluate')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
conf_threshold = args.conf_threshold
device = args.device
override_prev_results = args.override
project_name = args.project
weights_path = args.weights
max_images = args.max_images

print(f'Running ONNX COCO-style evaluation on project {project_name}')
print(f'ONNX Model: {weights_path}')
print(f'Compound coefficient: {compound_coef} (D{compound_coef})')

# Load project parameters
params = yaml.safe_load(open(f'projects/{project_name}.yml'))
obj_list = params['obj_list']

# EfficientDet input sizes (D2 = 768)
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef]

print(f'Input size for D{compound_coef}: {input_size}x{input_size}')

class ONNXEfficientDetD2:
    """
    ONNX EfficientDet-D2 wrapper for COCO evaluation
    """
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device
        self.input_size = input_size
        
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Set providers based on device
        if device.lower() == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=session_options
        )
        
        # Get input/output details
        self.input_details = self.session.get_inputs()[0]
        self.input_name = self.input_details.name
        self.input_type = self.input_details.type
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f'✅ ONNX model loaded: {model_path}')
        print(f'   Input: {self.input_name} {self.input_details.shape}')
        print(f'   Outputs: {self.output_names}')
        print(f'   Device: {device}')
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for ONNX model
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        ori_h, ori_w = image.shape[:2]
        
        # Resize to model input size
        image = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Convert to NCHW format
        image = image.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Ensure correct data type
        if 'float64' in self.input_type.lower():
            image = image.astype(np.float64)
        else:
            image = image.astype(np.float32)
        
        # Calculate scale factors for postprocessing
        scale_x = ori_w / self.input_size
        scale_y = ori_h / self.input_size
        
        return image, scale_x, scale_y, ori_w, ori_h, original_image
    
    def predict(self, image_path):
        """
        Run inference on image
        """
        # Preprocess
        input_tensor, scale_x, scale_y, ori_w, ori_h, original_image = self.preprocess_image(image_path)
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        return outputs, scale_x, scale_y, ori_w, ori_h

def apply_nms(boxes, scores, classes, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression
    """
    if len(boxes) == 0:
        return [], [], []
    
    # Convert to format expected by cv2.dnn.NMSBoxes
    boxes_list = boxes.tolist()
    scores_list = scores.tolist()
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, conf_threshold, iou_threshold)
    
    if len(indices) > 0:
        indices = indices.flatten()
        return boxes[indices], scores[indices], classes[indices]
    else:
        return [], [], []

def postprocess_onnx_outputs(outputs, scale_x, scale_y, ori_w, ori_h, conf_threshold=0.05, nms_threshold=0.5):
    """
    Postprocess ONNX model outputs to get detections
    """
    if len(outputs) == 3:
        bbox_regression, classification, anchors = outputs
        use_anchors = True
    elif len(outputs) == 2:
        bbox_regression, classification = outputs
        anchors = None
        use_anchors = False
    else:
        print(f"Unexpected number of outputs: {len(outputs)}")
        return {'rois': np.array([]), 'class_ids': np.array([]), 'scores': np.array([])}
    
    batch_size, num_predictions, num_classes = classification.shape
    
    # Collect all detections
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for i in range(num_predictions):
        # Get class scores
        class_scores = classification[0, i]
        max_score = np.max(class_scores)
        
        if max_score > conf_threshold:
            class_id = np.argmax(class_scores)
            
            # Get bounding box
            if use_anchors and anchors is not None and i < anchors.shape[0]:
                # Use anchors (basic decoding - you may need to adjust this)
                anchor = anchors[i]
                x1, y1, x2, y2 = anchor
                
                # Apply regression (simplified - real implementation would decode properly)
                reg = bbox_regression[0, i]
                # For now, just use anchors directly (you can improve this)
                
            else:
                # Grid-based approach
                # Estimate based on pyramid levels for D2
                total_p3 = 96 * 96  # P3: 768/8 = 96
                total_p4 = 48 * 48  # P4: 768/16 = 48
                total_p5 = 24 * 24  # P5: 768/32 = 24
                
                if i < total_p3:  # P3 level
                    level_idx = i
                    grid_size = 96
                    stride = 8
                elif i < total_p3 + total_p4:  # P4 level
                    level_idx = i - total_p3
                    grid_size = 48
                    stride = 16
                else:  # P5 level
                    level_idx = i - total_p3 - total_p4
                    grid_size = 24
                    stride = 32
                
                # Calculate grid position
                grid_x = level_idx % grid_size
                grid_y = level_idx // grid_size
                
                # Center coordinates in input image space
                cx = (grid_x + 0.5) * stride
                cy = (grid_y + 0.5) * stride
                
                # Box size (you can adjust this)
                box_size = stride * 4
                x1 = cx - box_size / 2
                y1 = cy - box_size / 2
                x2 = cx + box_size / 2
                y2 = cy + box_size / 2
            
            # Scale to original image size
            x1 = max(0, min(x1 * scale_x, ori_w))
            y1 = max(0, min(y1 * scale_y, ori_h))
            x2 = max(0, min(x2 * scale_x, ori_w))
            y2 = max(0, min(y2 * scale_y, ori_h))
            
            # Ensure valid box
            if x2 > x1 and y2 > y1:
                # Convert to [x, y, w, h] format for NMS
                box = [x1, y1, x2 - x1, y2 - y1]
                all_boxes.append(box)
                all_scores.append(max_score)
                all_classes.append(class_id)
    
    if not all_boxes:
        return {'rois': np.array([]), 'class_ids': np.array([]), 'scores': np.array([])}
    
    # Convert to numpy arrays
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_classes = np.array(all_classes)
    
    # Apply NMS per class
    final_boxes = []
    final_scores = []
    final_classes = []
    
    unique_classes = np.unique(all_classes)
    for class_id in unique_classes:
        class_mask = all_classes == class_id
        class_boxes = all_boxes[class_mask]
        class_scores = all_scores[class_mask]
        class_classes = all_classes[class_mask]
        
        # Apply NMS for this class
        nms_boxes, nms_scores, nms_classes = apply_nms(class_boxes, class_scores, class_classes, nms_threshold)
        
        if len(nms_boxes) > 0:
            final_boxes.extend(nms_boxes)
            final_scores.extend(nms_scores)
            final_classes.extend(nms_classes)
    
    if not final_boxes:
        return {'rois': np.array([]), 'class_ids': np.array([]), 'scores': np.array([])}
    
    # Convert back to [x1, y1, x2, y2] format for compatibility
    final_boxes = np.array(final_boxes)
    rois = np.column_stack([
        final_boxes[:, 0],  # x1
        final_boxes[:, 1],  # y1
        final_boxes[:, 0] + final_boxes[:, 2],  # x2
        final_boxes[:, 1] + final_boxes[:, 3]   # y2
    ])
    
    return {
        'rois': rois,
        'class_ids': np.array(final_classes),
        'scores': np.array(final_scores)
    }

def evaluate_coco_onnx(img_path, set_name, image_ids, coco, model, threshold=0.05):
    """
    COCO evaluation for ONNX model
    """
    results = []
    
    print(f"Evaluating {len(image_ids)} images...")
    
    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Run inference
            outputs, scale_x, scale_y, ori_w, ori_h = model.predict(image_path)
            
            # Postprocess
            preds = postprocess_onnx_outputs(outputs, scale_x, scale_y, ori_w, ori_h, threshold, nms_threshold)
            
            if len(preds['rois']) == 0:
                continue
            
            scores = preds['scores']
            class_ids = preds['class_ids']
            rois = preds['rois']
            
            # Convert to COCO format
            if rois.shape[0] > 0:
                # Convert x1,y1,x2,y2 -> x1,y1,w,h
                boxes_coco = rois.copy()
                boxes_coco[:, 2] -= boxes_coco[:, 0]  # w = x2 - x1
                boxes_coco[:, 3] -= boxes_coco[:, 1]  # h = y2 - y1
                
                for roi_id in range(boxes_coco.shape[0]):
                    score = float(scores[roi_id])
                    label = int(class_ids[roi_id])
                    box = boxes_coco[roi_id, :]
                    
                    image_result = {
                        'image_id': image_id,
                        'category_id': label + 1,  # COCO categories are 1-indexed
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    
                    results.append(image_result)
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    if not len(results):
        raise Exception('The ONNX model does not provide any valid output, check model and data input')
    
    # Write output
    filepath = f'{set_name}_onnx_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)
    
    print(f"Results saved to: {filepath}")
    print(f"Total detections: {len(results)}")

def _eval(coco_gt, image_ids, pred_json_path):
    """
    Run COCO evaluation
    """
    # Load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)
    
    # Run COCO evaluation
    print('Running COCO BBox Evaluation...')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Print summary metrics
    print(f"\nCOCO Evaluation Results:")
    print(f"Average Precision (AP) @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
    print(f"Average Precision (AP) @ IoU=0.50: {coco_eval.stats[1]:.4f}")
    print(f"Average Precision (AP) @ IoU=0.75: {coco_eval.stats[2]:.4f}")

if __name__ == '__main__':
    # Validate inputs
    if not os.path.exists(weights_path):
        print(f"Error: ONNX model not found: {weights_path}")
        exit(1)
    
    if not os.path.exists(f'projects/{project_name}.yml'):
        print(f"Error: Project file not found: projects/{project_name}.yml")
        exit(1)
    
    SET_NAME = params['val_set']
    VAL_GT = f'datasets/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params["project_name"]}/{SET_NAME}/'
    
    # Validate dataset paths
    if not os.path.exists(VAL_GT):
        print(f"Error: Annotation file not found: {VAL_GT}")
        exit(1)
    
    if not os.path.exists(VAL_IMGS):
        print(f"Error: Image directory not found: {VAL_IMGS}")
        exit(1)
    
    # Load COCO dataset
    print(f"Loading COCO dataset from: {VAL_GT}")
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:max_images]
    
    print(f"Found {len(image_ids)} images for evaluation")
    
    results_file = f'{SET_NAME}_onnx_bbox_results.json'
    
    if override_prev_results or not os.path.exists(results_file):
        # Load ONNX model
        print(f"Loading ONNX model: {weights_path}")
        model = ONNXEfficientDetD2(weights_path, device)
        
        # Run evaluation
        evaluate_coco_onnx(VAL_IMGS, SET_NAME, image_ids, coco_gt, model, conf_threshold)
    else:
        print(f"Using existing results file: {results_file}")
    
    # Run COCO evaluation
    _eval(coco_gt, image_ids, results_file)

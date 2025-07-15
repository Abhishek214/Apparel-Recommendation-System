# ONNX vs PyTorch Model Accuracy Comparison Script
import torch
import numpy as np
import onnxruntime as ort
import cv2
import json
import time
from tqdm import tqdm
import os
import argparse

# Import your existing evaluation components
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class ONNXEfficientDetEvaluator:
    """
    ONNX model evaluator that mimics PyTorch evaluation pipeline
    """
    
    def __init__(self, onnx_path, compound_coef=1, num_classes=4):
        self.onnx_path = onnx_path
        self.compound_coef = compound_coef
        self.num_classes = num_classes
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = self.input_sizes[compound_coef]
        
        # Load ONNX session
        print(f"Loading ONNX model: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Generate anchors (since ONNX model doesn't include them)
        self.anchors = self._generate_anchors()
        
        # Post-processing components
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        
        print(f"✅ ONNX evaluator initialized")
        print(f"   Input size: {self.input_size}")
        print(f"   Anchors shape: {self.anchors.shape}")
    
    def _generate_anchors(self):
        """Generate anchors for ONNX model (same as PyTorch)"""
        from efficientdet.utils import Anchors
        
        # Your anchor configuration
        anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        
        anchors_generator = Anchors(
            anchor_scale=anchor_scale[self.compound_coef],
            ratios=anchor_ratios,
            scales=anchor_scales
        )
        
        # Generate anchors for standard input size
        dummy_input = torch.zeros(1, 3, self.input_size, self.input_size)
        anchors = anchors_generator(dummy_input, dummy_input.dtype)
        
        return anchors
    
    def predict_single_image(self, image_path, threshold=0.2, nms_threshold=0.5):
        """
        Run inference on single image using ONNX model
        """
        # Preprocess image (same as PyTorch)
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=self.input_size)
        
        # Convert to ONNX input format
        x = np.stack([fi for fi in framed_imgs], axis=0)
        x = x.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC to NCHW, ensure float32
        
        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: x})
        regression, classification = outputs
        
        # Convert back to PyTorch tensors for post-processing
        regression = torch.from_numpy(regression)
        classification = torch.from_numpy(classification)
        
        # Post-process (same as PyTorch)
        out = postprocess(torch.from_numpy(x),
                         self.anchors, regression, classification,
                         self.regressBoxes, self.clipBoxes,
                         threshold=threshold, nms_threshold=nms_threshold)
        
        # Invert preprocessing transformations
        out = invert_affine(framed_metas, out)
        
        return out

class PyTorchEfficientDetEvaluator:
    """
    PyTorch model evaluator for comparison
    """
    
    def __init__(self, model_path, compound_coef=1, num_classes=4):
        self.compound_coef = compound_coef
        self.num_classes = num_classes
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.input_size = self.input_sizes[compound_coef]
        
        # Your anchor configuration
        anchor_ratios = [(0.7, 1.4), (1.0, 1.0), (1.5, 0.7)]
        anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        
        # Load PyTorch model
        print(f"Loading PyTorch model: {model_path}")
        self.model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=num_classes,
            ratios=anchor_ratios,
            scales=anchor_scales
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        self.model.eval()
        
        # Post-processing components
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        
        print(f"✅ PyTorch evaluator initialized")
    
    def predict_single_image(self, image_path, threshold=0.2, nms_threshold=0.5):
        """
        Run inference on single image using PyTorch model
        """
        # Preprocess image
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=self.input_size)
        
        # Convert to PyTorch tensors
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)
        
        # Run PyTorch inference
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)
        
        # Post-process
        out = postprocess(x, anchors, regression, classification,
                         self.regressBoxes, self.clipBoxes,
                         threshold=threshold, nms_threshold=nms_threshold)
        
        # Invert preprocessing transformations
        out = invert_affine(framed_metas, out)
        
        return out

def convert_predictions_to_coco_format(predictions, image_id, class_names):
    """
    Convert model predictions to COCO evaluation format
    """
    coco_results = []
    
    if len(predictions[0]['rois']) == 0:
        return coco_results
    
    rois = predictions[0]['rois']
    scores = predictions[0]['scores']
    class_ids = predictions[0]['class_ids']
    
    for i in range(len(rois)):
        x1, y1, x2, y2 = rois[i]
        
        coco_result = {
            'image_id': image_id,
            'category_id': int(class_ids[i]) + 1,  # COCO categories are 1-indexed
            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            'score': float(scores[i])
        }
        coco_results.append(coco_result)
    
    return coco_results

def evaluate_model_on_dataset(evaluator, dataset_path, annotation_file, model_name):
    """
    Evaluate model on validation dataset
    """
    print(f"\n🔍 Evaluating {model_name}...")
    
    # Load COCO annotations
    coco_gt = COCO(annotation_file)
    image_ids = list(coco_gt.imgs.keys())
    
    all_predictions = []
    inference_times = []
    
    # Run inference on all images
    for image_id in tqdm(image_ids, desc=f"Processing {model_name}"):
        image_info = coco_gt.imgs[image_id]
        image_path = os.path.join(dataset_path, image_info['file_name'])
        
        if not os.path.exists(image_path):
            continue
        
        # Measure inference time
        start_time = time.time()
        predictions = evaluator.predict_single_image(image_path)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # Convert to COCO format
        class_names = ['signature', 'barcode', 'chop', 'qrcode']  # Your classes
        coco_predictions = convert_predictions_to_coco_format(predictions, image_id, class_names)
        all_predictions.extend(coco_predictions)
    
    # Run COCO evaluation
    if len(all_predictions) > 0:
        coco_dt = coco_gt.loadRes(all_predictions)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        # Extract key metrics
        metrics = {
            'mAP_0.5_0.95': coco_eval.stats[0],
            'mAP_0.5': coco_eval.stats[1],
            'mAP_0.75': coco_eval.stats[2],
            'mAP_small': coco_eval.stats[3],
            'mAP_medium': coco_eval.stats[4],
            'mAP_large': coco_eval.stats[5],
            'avg_inference_time': np.mean(inference_times),
            'total_detections': len(all_predictions)
        }
    else:
        print("❌ No predictions generated")
        metrics = {key: 0.0 for key in ['mAP_0.5_0.95', 'mAP_0.5', 'mAP_0.75', 'mAP_small', 'mAP_medium', 'mAP_large']}
        metrics.update({'avg_inference_time': np.mean(inference_times) if inference_times else 0, 'total_detections': 0})
    
    return metrics

def compare_models(pytorch_path, onnx_path, dataset_path, annotation_file, compound_coef=1):
    """
    Compare PyTorch and ONNX model accuracy
    """
    print("🚀 Starting Model Comparison")
    print("="*60)
    
    # Initialize evaluators
    pytorch_evaluator = PyTorchEfficientDetEvaluator(pytorch_path, compound_coef, num_classes=4)
    onnx_evaluator = ONNXEfficientDetEvaluator(onnx_path, compound_coef, num_classes=4)
    
    # Evaluate PyTorch model
    pytorch_metrics = evaluate_model_on_dataset(
        pytorch_evaluator, dataset_path, annotation_file, "PyTorch"
    )
    
    # Evaluate ONNX model
    onnx_metrics = evaluate_model_on_dataset(
        onnx_evaluator, dataset_path, annotation_file, "ONNX"
    )
    
    # Print comparison results
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    print(f"{'Metric':<20} {'PyTorch':<12} {'ONNX':<12} {'Difference':<12}")
    print("-" * 60)
    
    for key in ['mAP_0.5_0.95', 'mAP_0.5', 'mAP_0.75', 'mAP_medium', 'mAP_large']:
        pytorch_val = pytorch_metrics[key]
        onnx_val = onnx_metrics[key]
        diff = onnx_val - pytorch_val
        
        status = "✅" if abs(diff) < 0.005 else "⚠️" if abs(diff) < 0.02 else "❌"
        
        print(f"{key:<20} {pytorch_val:<12.3f} {onnx_val:<12.3f} {diff:+.3f} {status}")
    
    # Performance comparison
    print(f"\n⏱️  Performance Comparison:")
    print(f"   PyTorch avg time: {pytorch_metrics['avg_inference_time']:.3f}s")
    print(f"   ONNX avg time:    {onnx_metrics['avg_inference_time']:.3f}s")
    speedup = pytorch_metrics['avg_inference_time'] / onnx_metrics['avg_inference_time']
    print(f"   ONNX speedup:     {speedup:.2f}x")
    
    # Detection count comparison
    print(f"\n📊 Detection Count:")
    print(f"   PyTorch detections: {pytorch_metrics['total_detections']}")
    print(f"   ONNX detections:    {onnx_metrics['total_detections']}")
    
    # Overall assessment
    max_diff = max(abs(onnx_metrics[key] - pytorch_metrics[key]) for key in ['mAP_0.5_0.95', 'mAP_0.5', 'mAP_0.75'])
    
    print(f"\n🎯 Overall Assessment:")
    if max_diff < 0.005:
        print("✅ EXCELLENT: ONNX model maintains accuracy (< 0.5% difference)")
    elif max_diff < 0.02:
        print("⚠️  GOOD: ONNX model has minor accuracy loss (< 2% difference)")
    else:
        print("❌ POOR: ONNX model has significant accuracy loss (> 2% difference)")
    
    return pytorch_metrics, onnx_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare PyTorch and ONNX model accuracy')
    parser.add_argument('--pytorch_model', type=str, required=True, help='Path to PyTorch .pth model')
    parser.add_argument('--onnx_model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to validation dataset')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to COCO annotation file')
    parser.add_argument('--compound_coef', type=int, default=1, help='EfficientDet compound coefficient')
    
    args = parser.parse_args()
    
    # Run comparison
    pytorch_metrics, onnx_metrics = compare_models(
        pytorch_path=args.pytorch_model,
        onnx_path=args.onnx_model,
        dataset_path=args.dataset_path,
        annotation_file=args.annotation_file,
        compound_coef=args.compound_coef
    )

# Usage example:
"""
python onnx_vs_pytorch_eval.py \
    --pytorch_model logs/abhi/efficientdet-d1_49_XXXX.pth \
    --onnx_model efficientdet_d1_stride_fixed.onnx \
    --dataset_path datasets/abhi/val2017 \
    --annotation_file datasets/abhi/annotations/instances_val2017.json \
    --compound_coef 1
"""

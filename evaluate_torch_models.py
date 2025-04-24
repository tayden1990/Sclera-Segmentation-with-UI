from PIL import Image
from tqdm import tqdm
import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import datetime
from pathlib import Path
import subprocess
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
# Import our project files
from utils import calculate_iou, calculate_precision_recall_f1, visualize_prediction
from torchvision.transforms import functional as F

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

def evaluate_model(model_path, test_image_dir, test_mask_dir, output_dir, threshold=0.7, size=512):
    """Evaluate a single model on all test images"""
    print(f"Evaluating model: {model_path}")
    
    # Set device with more verbose output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    # Load model
    try:
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 2 classes: background + sclera
        
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        print(f"Model device: {next(model.parameters()).device}")

        model.eval()
        print("Model loaded successfully to", device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    # Get list of test images
    test_images = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"Found {len(test_images)} test images")
    
    # Create output directory if it doesn't exist
    model_name = os.path.basename(model_path).split('.')[0]
    results_dir = os.path.join(output_dir, model_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize metrics
    iou_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    # Store detailed results
    detailed_results = []
    
    # Evaluate each image
    with tqdm(test_images, desc="Evaluating images") as pbar:
        for img_path in pbar:
            try:
                # Load and preprocess image
                img_filename = os.path.basename(img_path)
                mask_filename = img_filename.replace('.jpg', '.png').replace('.jpeg', '.png')
                mask_path = os.path.join(test_mask_dir, mask_filename)
                
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask not found for {img_filename}, skipping...")
                    continue
                
                # Load image and ground truth
                image = Image.open(img_path).convert('RGB')
                gt_mask = Image.open(mask_path).convert('L')
                
                # Resize image while preserving aspect ratio
                ratio = size / max(image.size)
                new_size = tuple([int(x * ratio) for x in image.size])
                image_resized = image.resize(new_size, Image.LANCZOS)
                
                # Convert to tensor
                img_tensor = F.to_tensor(image_resized).unsqueeze(0).to(device)
                print(f"Input tensor device: {img_tensor.device}")

                # Run inference
                with torch.no_grad():
                    prediction = model(img_tensor)
                
                # Process prediction
                if len(prediction[0]['masks']) > 0 and len(prediction[0]['scores']) > 0:
                    # Get highest scoring mask
                    scores = prediction[0]['scores']
                    best_idx = torch.argmax(scores)
                    score = scores[best_idx].item()
                    mask = prediction[0]['masks'][best_idx, 0]
                    
                    # Apply threshold
                    pred_mask = (mask > threshold).cpu().numpy()
                    
                    # Resize to original size
                    pred_mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(image.size, Image.NEAREST)
                    pred_mask = np.array(pred_mask_pil) > 127
                    
                    # Convert ground truth to numpy array
                    gt_mask_np = np.array(gt_mask) > 127
                    
                    # Calculate metrics
                    iou = calculate_iou(gt_mask_np, pred_mask)
                    precision, recall, f1 = calculate_precision_recall_f1(gt_mask_np, pred_mask)
                    
                    # Save visualization
                    vis_path = os.path.join(results_dir, f"{os.path.splitext(img_filename)[0]}_vis.jpg")
                    visualize_prediction(image, gt_mask_np, pred_mask, score, iou, f1, vis_path)
                    
                    iou_scores.append(iou)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    # Save detailed results
                    detailed_results.append({
                        'image': img_filename,
                        'iou': iou,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'score': score
                    })
                else:
                    # No masks detected
                    iou_scores.append(0)
                    precision_scores.append(0)
                    recall_scores.append(0)
                    f1_scores.append(0)
                    
                    # Save visualization with empty prediction
                    gt_mask_np = np.array(gt_mask) > 127
                    pred_mask = np.zeros_like(gt_mask_np)
                    vis_path = os.path.join(results_dir, f"{os.path.splitext(img_filename)[0]}_vis.jpg")
                    visualize_prediction(image, gt_mask_np, pred_mask, 0, 0, 0, vis_path)
                    
                    detailed_results.append({
                        'image': img_filename,
                        'iou': 0,
                        'precision': 0,
                        'recall': 0,
                        'f1': 0,
                        'score': 0
                    })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Calculate averages
    avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Print results
    print(f"\nEvaluation results for {os.path.basename(model_path)}:")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # Create summary results
    avg_results = {
        'model': os.path.basename(model_path),
        'avg_iou': avg_iou,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1
    }
    
    # Create DataFrame for detailed results
    detailed_df = pd.DataFrame(detailed_results)
    
    # Save results to CSV
    results_csv = os.path.join(results_dir, 'detailed_results.csv')
    try:
        detailed_df.to_csv(results_csv, index=False)
        print(f"Saved detailed results to {results_csv}")
    except Exception as e:
        print(f"Error saving detailed results: {e}")
    
    # Return average results and detailed DataFrame
    return avg_results, detailed_df

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate sclera segmentation models')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Directory containing model checkpoints')
    parser.add_argument('--test-dir', type=str, default='sclera_dataset/test', help='Directory containing test images and masks')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for mask prediction')
    parser.add_argument('--size', type=int, default=512, help='Size to resize images to')
    # Add to argument parser in main()
    parser.add_argument('--force-cuda', action='store_true', help='Force CUDA usage')

    args = parser.parse_args()
    
    # Fix for datetime.utcnow() deprecation warning
    print(f"Current Date and Time (UTC): {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User: {os.getlogin()}")
    
    # Setup directories
    test_image_dir = os.path.join(args.test_dir, 'images')
    test_mask_dir = os.path.join(args.test_dir, 'masks')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find checkpoints
    checkpoint_files = []
    if os.path.isdir(args.checkpoint_dir):
        for root, _, files in os.walk(args.checkpoint_dir):
            for file in files:
                if file.endswith('.pth'):
                    checkpoint_files.append(os.path.join(root, file))
    else:
        checkpoint_files = [args.checkpoint_dir]
    
    checkpoint_files = sorted(checkpoint_files)
    print(f"Found {len(checkpoint_files)} checkpoint files")
    print(f"Will evaluate on images in {test_image_dir}")
    
    # Evaluate each model
    all_results = []
    
    for checkpoint in checkpoint_files:
        # Evaluate model and get results
        avg_result, detailed_df = evaluate_model(
            checkpoint, 
            test_image_dir, 
            test_mask_dir, 
            args.output_dir,
            args.threshold,
            args.size
        )
        
        if avg_result is not None:
            all_results.append(avg_result)
    
    # Save summary results if we have any
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_dir, 'summary_results.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary results saved to {summary_path}")
        
        # Print comparative results
        print("\nResults Summary:")
        print(summary_df.to_string(index=False))
        
        # Find best model by F1 score
        best_idx = summary_df['avg_f1'].idxmax()
        best_model = summary_df.iloc[best_idx]
        print(f"\nBest model: {best_model['model']} with F1 score: {best_model['avg_f1']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
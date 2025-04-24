import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def calculate_iou(gt_mask, pred_mask):
    """Calculate Intersection over Union for binary masks"""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_precision_recall_f1(gt_mask, pred_mask):
    """Calculate precision, recall and F1 score for binary masks"""
    true_positive = np.logical_and(gt_mask, pred_mask).sum()
    false_positive = np.logical_and(~gt_mask, pred_mask).sum()
    false_negative = np.logical_and(gt_mask, ~pred_mask).sum()
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def visualize_prediction(image, gt_mask, pred_mask, score=None, iou=None, f1=None, output_path=None):
    """Create a visualization of prediction vs ground truth"""
    # Convert image to numpy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Ground truth mask
    gt_overlay = image.copy()
    if len(gt_overlay.shape) == 2:  # Grayscale image
        gt_overlay = cv2.cvtColor(gt_overlay, cv2.COLOR_GRAY2RGB)
    
    gt_overlay[gt_mask, 1] = 255  # Green for ground truth
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    # Prediction mask
    pred_overlay = image.copy()
    if len(pred_overlay.shape) == 2:  # Grayscale image
        pred_overlay = cv2.cvtColor(pred_overlay, cv2.COLOR_GRAY2RGB)
    
    pred_overlay[pred_mask, 0] = 255  # Red for prediction
    axes[2].imshow(pred_overlay)
    title = "Prediction"
    if score is not None:
        title += f" (Score: {score:.4f})"
    if iou is not None:
        title += f"\nIoU: {iou:.4f}"
    if f1 is not None:
        title += f", F1: {f1:.4f}"
    
    axes[2].set_title(title)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return fig
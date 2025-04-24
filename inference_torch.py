"""
Inference Script for Sclera Segmentation
----------------------------------------
Run inference on images using a trained model

Author: tayden1990
Created: 2025-04-23
"""
import os
import glob
import sys
import torch
import time
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

# Force CUDA visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

# Verify CUDA is available and print detailed info
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("PyTorch Version:", torch.__version__)
    print("CUDA Version:", torch.version.cuda)
    # Try to explicitly set device
    torch.cuda.set_device(0)
else:
    print("WARNING: CUDA is not available, using CPU")

def get_model_instance_segmentation(num_classes=2):
    """Create Mask R-CNN model with proper architecture"""
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Sclera Segmentation Inference')
    parser.add_argument('--weights', required=True, 
                        help='path to model weights (.pth file)')
    parser.add_argument('--image', required=True, 
                        help='path to image file or directory')
    parser.add_argument('--output', default='output', 
                        help='output directory (default: output)')
    parser.add_argument('--threshold', type=float, default=0.1, 
                        help='detection threshold (default: 0.1)')
    parser.add_argument('--resize', type=int, default=512, 
                        help='resize image to this size (default: 512)')
    parser.add_argument('--combine-masks', action='store_true', default=True,
                        help='combine multiple detections into one mask')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.weights, device)
    
    # Check if input is directory or file
    if os.path.isdir(args.image):
        # Process all images in directory
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(os.path.join(args.image, f"*.{ext}")))
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_file in image_files:
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            output_path = os.path.join(args.output, f"{base_name}")
            process_image(
                model, img_file, output_path, device, 
                args.threshold, args.resize, True, args.combine_masks
            )
    else:
        # Process single image
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_path = os.path.join(args.output, f"{base_name}")
        process_image(
            model, args.image, output_path, device,
            args.threshold, args.resize, True, args.combine_masks
        )

def process_image(model, image_path, output_path, device, threshold=0.1, resize=512, save_masks=True, combine_masks=True):
    """Process image with comprehensive visualization and metrics"""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path) or "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Load image
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            return False
            
        print(f"Processing image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return False
        
        # Convert to numpy for processing
        orig_image = np.array(image)
        orig_size = image.size  # (width, height)
        
        # Resize for model input
        if resize:
            image_resized = image.resize((resize, resize))
        else:
            image_resized = image
            
        # Convert to tensor
        image_tensor = F.to_tensor(image_resized).unsqueeze(0).to(device)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            predictions = model(image_tensor)
        inference_time = time.time() - start_time
            
        prediction = predictions[0]
        
        # Initialize metrics
        metrics = {
            "detections": 0,
            "max_confidence": 0.0,
            "avg_confidence": 0.0,
            "mask_coverage": 0.0,
            "inference_time": inference_time
        }
        
        # Check if any objects were detected
        if len(prediction['boxes']) > 0:
            # Get detections above threshold
            scores = prediction['scores']
            high_idx = torch.where(scores > threshold)[0]
            
            # Create combined mask for all detections
            height, width = orig_image.shape[:2]
            combined_mask = np.zeros((height, width), dtype=bool)
            
            # Process each detection above threshold
            if len(high_idx) > 0:
                metrics["detections"] = len(high_idx)
                metrics["max_confidence"] = scores[high_idx[0]].item()
                metrics["avg_confidence"] = scores[high_idx].mean().item()
                
                for i in high_idx:
                    # Get current detection
                    mask = prediction['masks'][i, 0].cpu().numpy()
                    
                    # Convert float mask to binary and resize to original image
                    binary_mask = mask > 0.5
                    resized_mask = cv2.resize(binary_mask.astype(np.uint8), 
                                             (orig_size[0], orig_size[1])) > 0
                    
                    # Add to combined mask
                    if combine_masks:
                        combined_mask = np.logical_or(combined_mask, resized_mask)
                
                # Post-process the mask to improve visualization
                if combine_masks:
                    # Apply morphological operations to clean up mask
                    kernel = np.ones((5, 5), np.uint8)
                    processed_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), 
                                                     cv2.MORPH_CLOSE, kernel)
                    processed_mask = cv2.morphologyEx(processed_mask, 
                                                     cv2.MORPH_OPEN, kernel)
                    combined_mask = processed_mask > 0
                
                # Calculate mask coverage
                metrics["mask_coverage"] = np.sum(combined_mask) / combined_mask.size * 100
                
                # 1. Save binary mask image
                binary_mask_path = os.path.join(output_dir, f"{base_filename}_binary_mask.png")
                cv2.imwrite(binary_mask_path, combined_mask.astype(np.uint8) * 255)
                print(f"Binary mask saved to: {binary_mask_path}")
                
                # 2. Create red overlay mask image
                cv_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)
                cv_overlay = cv_image.copy()
                
                # Make the mask overlay bright red
                cv_overlay[combined_mask, 0] = 0      # B channel (BGR format)
                cv_overlay[combined_mask, 1] = 0      # G channel
                cv_overlay[combined_mask, 2] = 255    # R channel
                
                # Create blended image
                cv_result = cv2.addWeighted(cv_image, 0.3, cv_overlay, 0.7, 0)
                
                # Save the red overlay
                red_overlay_path = os.path.join(output_dir, f"{base_filename}_red_overlay.png")
                cv2.imwrite(red_overlay_path, cv_result)
                print(f"Red overlay saved to: {red_overlay_path}")
                
                # 3. Create side-by-side comparison with metrics
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Original image
                ax1.imshow(orig_image)
                ax1.set_title("Original Image")
                ax1.axis('off')
                
                # Segmentation with red overlay
                red_overlay = orig_image.copy()
                red_overlay[combined_mask, 0] = 255  # R
                red_overlay[combined_mask, 1] = 0    # G
                red_overlay[combined_mask, 2] = 0    # B
                
                # Display overlay with metrics
                ax2.imshow(red_overlay)
                ax2.set_title("Sclera Segmentation")
                ax2.axis('off')
                
                # Add metrics text box
                metrics_text = "\n".join([
                    f"Detections: {metrics['detections']}",
                    f"Max Confidence: {metrics['max_confidence']:.3f}",
                    f"Avg Confidence: {metrics['avg_confidence']:.3f}",
                    f"Mask Coverage: {metrics['mask_coverage']:.1f}%",
                    f"Inference Time: {metrics['inference_time']:.3f}s"
                ])
                
                # Add text box with metrics
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                fig.text(0.5, 0.05, metrics_text, fontsize=12,
                        verticalalignment='bottom', horizontalalignment='center',
                        bbox=props)
                
                # Save the comparison figure
                comparison_path = os.path.join(output_dir, f"{base_filename}_comparison.png")
                plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Make room for metrics text
                plt.savefig(comparison_path, dpi=150)
                print(f"Comparison image saved to: {comparison_path}")
                plt.close(fig)
                
                return True
            
        # If no detection, save original image with metrics showing zero detections
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(orig_image)
        ax.set_title("No Sclera Detected")
        ax.axis('off')
        
        # Add metrics text box
        metrics_text = "\n".join([
            "Detections: 0",
            f"Max Confidence: 0.000",
            f"Avg Confidence: 0.000",
            f"Mask Coverage: 0.0%",
            f"Inference Time: {metrics['inference_time']:.3f}s"
        ])
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        fig.text(0.5, 0.05, metrics_text, fontsize=12,
                verticalalignment='bottom', horizontalalignment='center',
                bbox=props)
        
        # Save the result
        no_detection_path = os.path.join(output_dir, f"{base_filename}_no_detection.png")
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(no_detection_path, dpi=150)
        print(f"Original image saved to: {no_detection_path} (no detections)")
        plt.close(fig)
        return False
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_image_direct(model, image_path, output_path, device, threshold=0.5, resize=512, show=False, save_masks=True):
    """Process a single image with direct OpenCV visualization"""
    try:
        # Verify image exists
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}")
            return False
            
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image: {image_path}")
            return False
            
        # Convert to RGB for processing
        orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_size = (orig_image.shape[1], orig_image.shape[0])
        
        # Create PIL image for PyTorch processing
        pil_image = Image.fromarray(orig_image)
        
        # Resize for model input
        if resize:
            pil_image_resized = pil_image.resize((resize, resize))
        else:
            pil_image_resized = pil_image
            
        # Convert to tensor and move to device
        image_tensor = F.to_tensor(pil_image_resized).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(image_tensor)
            
        prediction = predictions[0]
        
        # Check if any masks are found
        mask_found = False
        combined_mask = np.zeros(orig_image.shape[:2], dtype=np.uint8)
        
        if len(prediction['masks']) > 0:
            # Get high confidence masks
            scores = prediction['scores']
            high_idx = torch.where(scores > threshold)[0]
            
            if len(high_idx) > 0:
                # Process each detected mask
                for i in high_idx:
                    # Get mask and resize to original image
                    mask = prediction['masks'][i, 0].cpu().numpy()
                    mask = mask > 0.5
                    mask_resized = cv2.resize(mask.astype(np.uint8), 
                                             (orig_size[0], orig_size[1]))
                    
                    # Add to combined mask
                    combined_mask = np.maximum(combined_mask, mask_resized)
                    
                    # Get bounding box
                    box = prediction['boxes'][i].cpu().numpy().astype(int)
                    score = scores[i].item()
                    
                    # Draw bounding box
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    
                    # Add score text
                    cv2.putText(image, f"{score:.2f}", (box[0], box[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Create colored mask
                colored_mask = np.zeros_like(image)
                colored_mask[combined_mask > 0] = [0, 0, 255]  # Red in BGR
                
                # Blend with original image
                alpha = 0.6
                mask_overlay = cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)
                
                # Set flag for mask found
                mask_found = True
                
                # Save result
                cv2.imwrite(output_path, mask_overlay)
                print(f"Result saved to: {output_path}")
                
                # Save separate binary mask if requested
                if save_masks:
                    mask_path = output_path.replace('.png', '_mask.png')
                    mask_path = mask_path.replace('.jpg', '_mask.png')
                    mask_path = mask_path.replace('.jpeg', '_mask.png')
                    cv2.imwrite(mask_path, combined_mask * 255)
                    print(f"Mask saved to: {mask_path}")
                
                # Display if requested
                if show:
                    cv2.imshow("Original", image)
                    cv2.imshow("Segmentation Result", mask_overlay)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                return True
            
        # No mask found
        if not mask_found:
            print(f"No sclera detected in {image_path}")
            cv2.imwrite(output_path, image)
            print(f"Original image saved to: {output_path}")
            
            if show:
                cv2.imshow("Original (No detection)", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
            return False
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_model(weights_path, device):
    """Load Mask R-CNN model with proper initialization"""
    # Create model with correct architecture
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    
    # Initialize with pretrained weights
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # Background + sclera
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Check if this is a training checkpoint or direct state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        # Move model to device and set to evaluation mode
        model.to(device)
        model.eval()
        print(f"Model loaded successfully and moved to {device}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def analyze_model_performance(model, device, test_dir, threshold=0.5, image_size=512):
    """Analyze model performance on a test directory"""
    print("\n=== Model Performance Analysis ===")
    
    # Find test images
    import glob
    image_files = glob.glob(os.path.join(test_dir, "images", "*.jpg"))
    image_files += glob.glob(os.path.join(test_dir, "images", "*.png"))
    
    # Find corresponding masks
    mask_dir = os.path.join(test_dir, "masks")
    
    if not os.path.exists(mask_dir):
        print(f"Mask directory not found: {mask_dir}")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # Track metrics
    scores = []
    ious = []
    
    # Process each image
    for image_file in image_files:
        try:
            # Get base filename
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            
            # Find corresponding mask file
            mask_file = None
            for ext in ['.png', '.jpg']:
                possible_mask = os.path.join(mask_dir, base_name + ext)
                if os.path.exists(possible_mask):
                    mask_file = possible_mask
                    break
            
            if not mask_file:
                print(f"No mask found for {image_file}")
                continue
                
            # Load and preprocess image
            image = Image.open(image_file).convert("RGB")
            
            # Resize for model input
            if image_size:
                image_input = F.resize(image, (image_size, image_size))
            else:
                image_input = image
                
            # Run inference
            image_tensor = F.to_tensor(image_input).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(image_tensor)
                
            # Process predictions
            prediction = predictions[0]
            
            # Check if any masks are found
            if len(prediction['masks']) > 0:
                # Get highest scoring mask
                pred_scores = prediction['scores']
                high_idx = torch.where(pred_scores > threshold)[0]
                
                if len(high_idx) > 0:
                    # Get binary mask
                    pred_mask = prediction['masks'][high_idx[0], 0].cpu().numpy()
                    pred_mask = pred_mask > 0.5
                    
                    # Record score
                    score = pred_scores[high_idx[0]].item()
                    scores.append(score)
                    
                    # Load ground truth mask
                    gt_mask = np.array(Image.open(mask_file).convert("L"))
                    gt_mask = gt_mask > 127  # Convert to binary
                    
                    # Make sure masks are the same size for comparison
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                             (gt_mask.shape[1], gt_mask.shape[0]))
                        pred_mask = pred_mask > 0.5
                    
                    # Calculate IoU
                    intersection = np.logical_and(pred_mask, gt_mask).sum()
                    union = np.logical_or(pred_mask, gt_mask).sum()
                    iou = intersection / max(union, 1)
                    ious.append(iou)
                    
        except Exception as e:
            print(f"Error analyzing {image_file}: {e}")
            
    # Report results
    if scores:
        print(f"Average detection score: {sum(scores)/len(scores):.4f}")
    if ious:
        print(f"Average IoU: {sum(ious)/len(ious):.4f}")
        
    return scores, ious

# Add this function to help with path normalization
def normalize_path(path):
    """Normalize path for Windows compatibility"""
    # Remove quotes if they exist
    path = path.strip('\'"')
    # Convert to absolute path
    path = os.path.abspath(path)
    # Normalize slashes
    path = os.path.normpath(path)
    return path

def visualize_instances(image, boxes, masks, class_ids, class_names, scores, ax=None):
    """
    Visualize instances similar to mrcnn.visualize.display_instances
    """
    # Create axis if none provided
    if ax is None:
        _, ax = plt.subplots(1, figsize=(10, 10))
    
    # Show image
    ax.imshow(image)
    
    # Generate random colors for each instance
    import random
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
             for _ in range(len(boxes))]
    
    # Process each instance
    for i, (box, mask, score, color) in enumerate(zip(boxes, masks, scores, colors)):
        # Convert color to 0-1 range for matplotlib
        color_norm = [c / 255.0 for c in color]
        
        # Draw bounding box
        y1, x1, y2, x2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                           edgecolor=color_norm, facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        
        # Draw label
        class_id = class_ids[i]
        class_name = class_names[class_id]
        label = f"{class_name}: {score:.3f}"
        ax.text(x1, y1, label, color='white', 
               bbox=dict(facecolor=color_norm, alpha=0.7))
        
        # Apply color mask
        masked_image = np.zeros_like(image.astype(np.uint8))
        for c in range(3):
            masked_image[:, :, c] = np.where(mask, 
                                          color_norm[c] * 255, 
                                          masked_image[:, :, c])
        
        # Blend with original image
        ax.imshow(masked_image, alpha=0.5)
    
    ax.axis('off')
    return ax

if __name__ == "__main__":
    main()
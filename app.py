"""
Sclera Segmentation Gradio Interface
------------------------------------
An interactive web UI for training, inference, and evaluation
of sclera segmentation models using PyTorch and Gradio.

Author: tayden1990
Created: 2025-04-23
"""
import os
import sys
import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import subprocess
import glob
import pandas as pd
from datetime import datetime
import json
import cv2
from pathlib import Path
import threading
import time
from torchvision.transforms import functional as F

# Import our model handling code
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, maskrcnn_resnet50_fpn

# Define directory constants
DATASET_DIR = os.path.join(os.path.dirname(__file__), "sclera_dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "logs")
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Create directories if they don't exist
for dir_path in [DATASET_DIR, MODEL_DIR, TEMP_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Global variable to track training status
training_status = {
    "active": False,
    "message": "",
    "progress": 0,
    "log": ""
}

def find_checkpoints():
    """Find available checkpoint directories in the model directory"""
    if not os.path.exists(MODEL_DIR):
        return []  # Return empty list if directory doesn't exist
        
    checkpoints = [d for d in os.listdir(MODEL_DIR) 
                  if os.path.isdir(os.path.join(MODEL_DIR, d))]
    return sorted(checkpoints)

def find_model_files(model_dir):
    """Find available model files in a specific checkpoint directory"""
    if not model_dir:  # Handle empty selection
        return []
        
    # Handle case where model_dir is a list (happens with refresh)
    if isinstance(model_dir, list):
        return []
        
    model_path = os.path.join(MODEL_DIR, model_dir)
    if not os.path.exists(model_path):
        return []  # Return empty list if directory doesn't exist
    
    # Look for checkpoint files and final model
    model_files = [f for f in os.listdir(model_path)
                  if f.endswith('.pth')]
    
    return sorted(model_files)

def load_model(model_path):
    """Load a PyTorch model from the given path"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 2 classes: background and sclera
    
    # Replace mask predictor
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
    model.eval()
    
    return model, device

def preprocess_image(image, size=512):
    """Preprocess an image for inference"""
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image))
    
    # Store original size
    orig_size = image.size
    
    # Resize image
    image = image.resize((size, size))
    
    # Convert to tensor
    image_tensor = F.to_tensor(image).unsqueeze(0)
    
    return image_tensor, orig_size

def predict_mask(model, image_tensor, device, threshold=0.5):
    """Run inference and return the predicted mask"""
    # Move tensor to device
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = model(image_tensor)[0]
    
    # Get highest scoring mask above threshold
    if len(prediction['masks']) > 0 and len(prediction['scores']) > 0:
        scores = prediction['scores']
        high_scores = torch.where(scores > threshold)[0]
        
        if len(high_scores) > 0:
            highest_idx = high_scores[0]
            mask = prediction['masks'][highest_idx, 0].cpu().numpy()
            mask = mask > 0.5
            score = scores[highest_idx].item()
            return mask, score
    
    # No mask detected
    return None, 0.0

def visualize_result(image, mask, score, orig_size=None):
    """Visualize the segmentation result on the image"""
    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = np.array(image)
    
    # If no mask detected
    if mask is None:
        return Image.fromarray(image_np)
    
    # Resize mask to match image if needed
    if orig_size is not None and (mask.shape[0] != orig_size[1] or mask.shape[1] != orig_size[0]):
        mask_resized = cv2.resize(mask.astype(np.uint8), (orig_size[0], orig_size[1]))
        mask = mask_resized > 0
    
    # Create overlay
    overlay = image_np.copy()
    overlay[mask, 0] = 255  # Red channel
    
    # Blend images
    alpha = 0.5
    result = cv2.addWeighted(image_np, 1, overlay, alpha, 0)
    
    # Add text for score
    if score > 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Score: {score:.4f}", (10, 30), font, 0.8, (0, 0, 255), 2)
    
    return Image.fromarray(result)

def create_training_tab():
    """Create the training tab with detailed visual progress"""
    with gr.Tab("Train Model"):
        gr.Markdown("""
        # Train a New Sclera Segmentation Model
        
        This tab lets you train a new model with custom parameters.
        The training process may take a long time depending on your dataset size and hardware.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                epochs = gr.Slider(minimum=1, maximum=100, value=5, step=1, label="Number of Epochs")
                batch_size = gr.Slider(minimum=1, maximum=16, value=2, step=1, label="Batch Size")
                image_size = gr.Slider(minimum=128, maximum=512, value=256, step=32, label="Image Size")
                subset = gr.Slider(minimum=0.01, maximum=1.0, value=0.1, step=0.01, label="Dataset Subset")
                save_dir = gr.Textbox(label="Model Name", value="sclera_model")
                
                train_button = gr.Button("Start Training", variant="primary")
            
            with gr.Column(scale=2):
                # Training visualization area
                with gr.Group():
                    status_indicator = gr.Textbox(label="Status", value="Not started")
                    epoch_progress = gr.HTML(value="<div style='text-align:center;'>Epoch: 0/0</div>")
                    progress_bar = gr.HTML(value="""
                        <div style="width:100%; background-color:#ddd; border-radius:5px;">
                            <div style="width:0%; height:24px; background-color:#4CAF50; border-radius:5px; text-align:center; line-height:24px; color:white;">
                                0%
                            </div>
                        </div>
                    """)
                    
                train_log = gr.Textbox(label="Training Log", lines=20, autoscroll=True, value="")
                
                # Add image output to show training curve
                training_plot = gr.Plot(label="Training Progress")
        
        # File to store real-time log
        log_file_path = os.path.join(TEMP_DIR, "current_training_log.txt")
        
        # Function to start training process
        def start_training(epochs, batch_size, image_size, subset, save_dir):
            # Clear previous log file
            os.makedirs(TEMP_DIR, exist_ok=True)
            with open(log_file_path, "w") as f:
                f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Parameters: Epochs={epochs}, Batch Size={batch_size}, Image Size={image_size}, Subset={subset}\n\n")
            
            # Start training in background thread
            def run_training():
                # Get the path to the Python interpreter in the virtual environment
                python_exe = sys.executable  # This gets the current Python interpreter path
                
                # Determine which training script exists
                train_script = "train.py"
                if not os.path.exists(os.path.join(os.path.dirname(__file__), train_script)):
                    train_script = "train_noskimage.py"
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cmd = [
                    python_exe,  # Use the same Python interpreter as running app.py
                    train_script,
                    "--epochs", str(epochs),
                    "--batch-size", str(batch_size),
                    "--image-size", str(image_size),
                    "--subset", str(subset),
                    "--log-dir", f"logs/{save_dir}_{timestamp}"
                ]
                
                # Print command for debugging
                print("Running command:", " ".join(cmd))
                
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    bufsize=1
                )
                
                # Variables to track progress
                current_epoch = 0
                total_epochs = epochs
                
                # Process output in real-time
                for line in process.stdout:
                    # Append to log file
                    with open(log_file_path, "a") as f:
                        f.write(line)
                    
                    # Update progress variables
                    if "Epoch" in line and "/" in line:
                        try:
                            parts = line.split("/")
                            if len(parts) >= 2:
                                current_epoch = int(parts[0].split()[-1])
                        except Exception as e:
                            print(f"Error parsing epoch: {e}")
                    
                    # Print to console for debugging
                    print(line.strip())
                
                # Wait for process to complete
                process.wait()
                
                # Mark training as completed
                with open(log_file_path, "a") as f:
                    if process.returncode == 0:
                        f.write("\n\n===== Training completed successfully! =====\n")
                    else:
                        f.write(f"\n\n===== Training failed with error code {process.returncode} =====\n")
            
            # Start the thread
            thread = threading.Thread(target=run_training)
            thread.daemon = True
            thread.start()
            
            return "Training started. See log for details."
        
        # Function to update UI with current progress
        def update_training_status():
            if not os.path.exists(log_file_path):
                return (
                    "Not started",
                    "<div style='text-align:center;'>Epoch: 0/0</div>",
                    """<div style="width:100%; background-color:#ddd; border-radius:5px;">
                        <div style="width:0%; height:24px; background-color:#4CAF50; border-radius:5px; text-align:center; line-height:24px; color:white;">
                            0%
                        </div>
                    </div>""",
                    "",
                    None
                )
            
            try:
                with open(log_file_path, "r") as f:
                    log_content = f.read()
                
                # Parse log to extract progress
                current_epoch = 0
                total_epochs = 0
                
                for line in log_content.split("\n"):
                    if "Epoch" in line and "/" in line:
                        try:
                            parts = line.split("/")
                            if len(parts) >= 2:
                                current_epoch = int(parts[0].split()[-1])
                                total_epochs = int(parts[1].split()[0])
                        except Exception as e:
                            print(f"Error parsing line: {line}, {e}")
                
                # Calculate progress percentage
                progress_pct = 0 if total_epochs == 0 else min(100, (current_epoch / total_epochs) * 100)
                
                # Generate progress bar HTML
                progress_html = f"""
                <div style="width:100%; background-color:#ddd; border-radius:5px;">
                    <div style="width:{progress_pct}%; height:24px; background-color:#4CAF50; border-radius:5px; text-align:center; line-height:24px; color:white;">
                        {progress_pct:.1f}%
                    </div>
                </div>
                """
                
                # Generate epoch progress HTML
                epoch_html = f"<div style='text-align:center;'>Epoch: {current_epoch}/{total_epochs}</div>"
                
                # Determine status message
                if "Training completed successfully" in log_content:
                    status = "Completed Successfully"
                elif "Training failed" in log_content:
                    status = "Failed"
                else:
                    status = "Running"
                
                # Create a simple plot for visualization
                fig = None
                if current_epoch > 0:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(['Progress'], [progress_pct], color='green')
                    ax.set_ylim(0, 100)
                    ax.set_ylabel('Completion (%)')
                    ax.set_title(f'Training Progress - Epoch {current_epoch}/{total_epochs}')
                
                return status, epoch_html, progress_html, log_content, fig
                
            except Exception as e:
                print(f"Error updating status: {str(e)}")
                return "Error", "<div>Error reading progress</div>", "<div>Error</div>", f"Error: {str(e)}", None
        
        # Connect the button to start training
        train_button.click(
            fn=start_training,
            inputs=[epochs, batch_size, image_size, subset, save_dir],
            outputs=[status_indicator]
        )
        
        # Refresh button for manual status update
        refresh_button = gr.Button("Refresh Status", variant="secondary")
        refresh_button.click(
            fn=update_training_status,
            inputs=[],
            outputs=[status_indicator, epoch_progress, progress_bar, train_log, training_plot]
        )

        # Add automatic refresh for newer Gradio versions that support it
        try:
            # For Gradio 3.x
            refresh_button.click(
                fn=lambda: None,
                inputs=[],
                outputs=[],
                every=1  # Refresh every 1 second
            )
        except Exception:
            try:
                # For newer Gradio versions
                gr.Blocks.load(
                    fn=update_training_status,
                    inputs=[],
                    outputs=[status_indicator, epoch_progress, progress_bar, train_log, training_plot],
                    every=1
                )
            except Exception as e:
                print(f"Note: Automatic refresh not available in this Gradio version. Use the refresh button. Error: {e}")

def create_inference_tab():
    """Create the inference tab in the UI"""
    with gr.Tab("Inference"):
        gr.Markdown("## Run Inference on Images")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input image
                input_image = gr.Image(
                    type="pil", 
                    label="Input Image",
                    height=300
                )
                
                # Model selection
                available_checkpoints = find_checkpoints()
                model_dir = gr.Dropdown(
                    label="Model Directory",
                    choices=available_checkpoints,
                    interactive=True,
                    allow_custom_value=True
                )
                
                # Replace dropdown with textbox for model file
                model_file = gr.Textbox(
                    label="Model File (e.g., final_model.pth)",
                    placeholder="Enter model filename",
                    interactive=True
                )
                
                # Button to refresh available models
                refresh_btn = gr.Button("List Model Files")
                
                # Inference parameters
                threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Detection Threshold"
                )
                
                image_size = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Image Size"
                )
                
                # Run inference button
                infer_btn = gr.Button("Run Inference")
                
            with gr.Column(scale=1):
                # Output
                output_image = gr.Image(
                    type="pil",
                    label="Output Image",
                    height=300
                )
                
                result_text = gr.Textbox(
                    label="Results",
                    interactive=False
                )
        
        # Function to list model files
        def list_model_files(model_dir):
            """List model files in the selected directory"""
            # Handle case where model_dir is a list
            if isinstance(model_dir, list):
                model_dir = model_dir[0] if model_dir else None
            
            if not model_dir:
                return "Please select a model directory first"
            
            # Full path to model directory
            dir_path = os.path.join(MODEL_DIR, model_dir)
            
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # Get all .pth files in directory
                model_files = [f for f in os.listdir(dir_path) if f.endswith('.pth')]
                if model_files:
                    return f"Available model files: {', '.join(model_files)}"
                else:
                    return f"No model files found in {model_dir}"
            else:
                return f"Directory not found: {dir_path}"
        
        # Connect refresh button to list model files
        refresh_btn.click(
            fn=list_model_files,
            inputs=[model_dir],
            outputs=[result_text]
        )
        
        # Inference function
        def inference_single(image, model_dir, model_file, threshold=0.5, image_size=512):
            """Run inference on a single image"""
            if image is None:
                return None, "Please upload an image"
                
            if model_dir is None or not model_file:
                return None, "Please select both a model directory and enter a model file"
            
            # Handle cases where model_dir is a list
            if isinstance(model_dir, list):
                model_dir = model_dir[0] if model_dir else None
            
            # Handle case where model_file is a list
            if isinstance(model_file, list):
                model_file = model_file[0] if model_file else None
            
            # Construct full model path
            model_path = os.path.join(MODEL_DIR, model_dir, model_file)
            if not os.path.exists(model_path):
                return None, f"Model file not found: {model_path}"
            
            try:
                # Load model
                model, device = load_model(model_path)
                
                # Preprocess image 
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                
                if image.mode != "RGB":
                    image = image.convert("RGB")
                
                # Resize while preserving aspect ratio
                original_size = image.size  # PIL returns (width, height)
                ratio = image_size / max(original_size)
                new_size = tuple([int(x * ratio) for x in original_size])
                resized_img = image.resize(new_size, Image.LANCZOS)
                
                # Convert to tensor AND move to the same device as the model
                img_tensor = F.to_tensor(resized_img).unsqueeze(0).to(device)
                
                # Run inference
                with torch.no_grad():
                    prediction = model(img_tensor)
                    
                # Process prediction
                if len(prediction[0]['masks']) > 0:
                    # Get highest scoring mask
                    scores = prediction[0]['scores']
                    if len(scores) > 0:
                        # Get best mask (highest score)
                        best_idx = torch.argmax(scores)
                        best_score = scores[best_idx].item()
                        best_mask = prediction[0]['masks'][best_idx, 0]
                        
                        # Apply threshold
                        binary_mask = best_mask > threshold
                        
                        # Resize mask back to original size
                        mask_np = binary_mask.cpu().numpy().astype(np.uint8) * 255
                        mask_pil = Image.fromarray(mask_np)
                        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
                        
                        # Convert image to numpy for processing
                        image_np = np.array(image)
                        
                        # Create a binary mask image (white sclera, black background)
                        # Use the same dimensions as the image_np array
                        binary_result = np.zeros_like(image_np)
                        
                        # Convert mask to numpy array with correct dimensions
                        mask_np = np.array(mask_pil) > 0
                        
                        # Make sure dimensions match before indexing
                        if mask_np.shape[:2] != binary_result.shape[:2]:
                            # Resize mask to match binary_result dimensions
                            mask_np = np.array(Image.fromarray(mask_np.astype(np.uint8) * 255).resize(
                                (binary_result.shape[1], binary_result.shape[0]), 
                                Image.NEAREST
                            )) > 0
                        
                        # Apply the mask to create white sclera areas
                        binary_result[mask_np] = [255, 255, 255]
                        binary_img = Image.fromarray(binary_result)
                        
                        # Create red overlay
                        overlay_color = image_np.copy()
                        if len(overlay_color.shape) == 3:  # RGB image
                            overlay_color[mask_np, 0] = 255  # Red channel
                            overlay_color[mask_np, 1] = 0    # Green channel
                            overlay_color[mask_np, 2] = 0    # Blue channel
                        else:  # Grayscale image
                            overlay_color = np.stack([image_np] * 3, axis=2)
                            overlay_color[mask_np, 0] = 255  # Red channel
                            overlay_color[mask_np, 1] = 0    # Green channel
                            overlay_color[mask_np, 2] = 0    # Blue channel
                        
                        # Blend with original
                        alpha = 0.5
                        result = (1-alpha) * image_np + alpha * overlay_color
                        result = result.astype(np.uint8)
                        
                        # Convert back to PIL
                        result_img = Image.fromarray(result)
                        
                        # Create a combined visualization with both results side by side
                        combined = Image.new('RGB', (original_size[0] * 2, original_size[1]))
                        combined.paste(result_img, (0, 0))
                        combined.paste(binary_img, (original_size[0], 0))
                        
                        return combined, f"Inference completed. Detection score: {best_score:.4f}"
                    else:
                        return image, "No detection found with sufficient confidence"
                else:
                    return image, "No sclera detected in the image"
                    
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                return None, f"Error during inference: {str(e)}\n\nDetails:\n{error_details}"
        
        infer_btn.click(
            fn=inference_single,
            inputs=[input_image, model_dir, model_file, threshold, image_size],
            outputs=[output_image, result_text]
        )

def create_evaluation_tab():
    """Create the evaluation tab in the UI"""
    with gr.Tab("Evaluation"):
        gr.Markdown("## Evaluate Model Performance")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                available_checkpoints = find_checkpoints()
                model_dir = gr.Dropdown(
                    label="Model Directory",
                    choices=available_checkpoints,
                    value=None,
                    allow_custom_value=True,
                    interactive=True
                )
                
                # Test dataset selection
                test_dir = gr.Textbox(
                    label="Test Dataset Directory",
                    value="sclera_dataset/test",
                    interactive=True
                )
                
                # Evaluation parameters
                threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="Detection Threshold"
                )
                
                # Output directory
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="evaluation_results",
                    interactive=True
                )
                
                evaluate_btn = gr.Button("Evaluate Model")
                
            with gr.Column(scale=1):
                output_text = gr.Textbox(label="Evaluation Status", interactive=False)
                
                # Results display
                results_md = gr.Markdown("### Evaluation Results")
                results_df = gr.Dataframe(label="Results Summary", interactive=False)
        
        # Visualization section with simpler approach
        gr.Markdown("### Visualization Results")
        
        with gr.Row():
            # Simple text input for model name
            model_input = gr.Textbox(
                label="Enter Model Name (e.g., checkpoint_001)",
                placeholder="Enter model name here",
                interactive=True
            )
            
            # Button to load visualizations
            load_vis_btn = gr.Button("Load Visualizations")
        
        # Container for visualizations - only keep the gallery
        with gr.Row():
            gallery = gr.Gallery(
                label="Result Visualizations",
                show_label=True,
                elem_id="result_gallery",
                columns=3,
                object_fit="contain",
                height="auto"
            )
        
        # Run evaluation
        def run_evaluation(model_dir, test_dir, threshold, output_dir):
            if not model_dir:
                return "Please select a model directory", None
                
            model_path = os.path.join(MODEL_DIR, model_dir)
            
            # Build command
            cmd = [
                sys.executable,
                "evaluate_torch_models.py",
                "--checkpoint-dir", model_path,
                "--test-dir", test_dir,
                "--output-dir", output_dir,
                "--threshold", str(threshold),
                "--force-cuda"  # Force CUDA usage
            ]
            
            # Run evaluation script
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    return f"Evaluation failed: {stderr}", None
                
                # Check if results exist
                summary_path = os.path.join(output_dir, 'summary_results.csv')
                if not os.path.exists(summary_path):
                    return f"Evaluation completed but no summary results found at {summary_path}.\n\n{stdout}", None
                
                # Load summary results
                summary_df = pd.read_csv(summary_path)
                
                # Create a string with model names to display to the user
                model_names = [model.split('.')[0] for model in summary_df['model']]
                model_list_str = "Available models: " + ", ".join(model_names)
                
                # Return results and model list as a string
                return f"Evaluation completed successfully!\n\n{model_list_str}\n\n{stdout}", summary_df
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                return f"Error running evaluation: {str(e)}\n\n{error_details}", None
        
        evaluate_btn.click(
            fn=run_evaluation,
            inputs=[model_dir, test_dir, threshold, output_dir],
            outputs=[output_text, results_df]
        )
        
        # Function to load visualizations for a model - simplified to only return images
        def load_visualizations(model_name, output_dir):
            """Load visualizations for the entered model name"""
            if not model_name or not output_dir:
                return []
            
            print(f"Loading visualizations for model: {model_name}")
            print(f"Output directory: {output_dir}")
            
            # Make sure model_name is a string
            if isinstance(model_name, list):
                model_name = model_name[0] if model_name else ""
                
            # Strip any whitespace
            model_name = model_name.strip()
            
            results_dir = os.path.join(output_dir, model_name)
            if not os.path.exists(results_dir):
                print(f"Results directory not found: {results_dir}")
                return []
            
            # Find visualization images
            vis_images = [f for f in os.listdir(results_dir) if f.endswith('_vis.jpg')]
            if not vis_images:
                print(f"No visualization images found in {results_dir}")
                return []
            
            # Load images with their captions
            images = []
            for img_file in sorted(vis_images):
                img_path = os.path.join(results_dir, img_file)
                try:
                    img = Image.open(img_path)
                    img_name = img_file.replace('_vis.jpg', '')
                    images.append((img, img_name))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            
            print(f"Loaded {len(images)} images")
            return images
        
        # Connect the load button to the visualization function
        load_vis_btn.click(
            fn=load_visualizations,
            inputs=[model_input, output_dir],
            outputs=[gallery]
        )

def get_system_info():
    """Get real-time system information with fallback for missing dependencies"""
    import platform
    
    # Base info with always-available system information
    info = {
        # Python environment
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        
        # Deep learning framework
        "torch_version": torch.__version__,
        "cuda_available": "Yes" if torch.cuda.is_available() else "No",
    }
    
    # Add GPU information if available
    if torch.cuda.is_available():
        info["gpu_device"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        try:
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated() / (1024**3):.2f} GB"
            info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved() / (1024**3):.2f} GB"
        except:
            info["gpu_memory"] = "Error retrieving GPU memory information"
    
    # Try to get psutil information if available
    try:
        import psutil
        info["cpu_usage"] = f"{psutil.cpu_percent()}%"
        info["ram_used"] = f"{psutil.virtual_memory().percent}% ({psutil.virtual_memory().used / (1024**3):.2f} GB / {psutil.virtual_memory().total / (1024**3):.2f} GB)"
        
        # Use os.path.abspath('.') for Windows compatibility
        import os
        current_path = os.path.abspath('.')
        info["disk_usage"] = f"{psutil.disk_usage(current_path).percent}% ({psutil.disk_usage(current_path).used / (1024**3):.2f} GB / {psutil.disk_usage(current_path).total / (1024**3):.2f} GB)"
    except ImportError:
        info["system_resources"] = "Install psutil for CPU, RAM, and disk usage (pip install psutil)"
    
    # Try to get GPU utilization if nvidia-smi is available
    if torch.cuda.is_available():
        try:
            import subprocess
            result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
            gpu_util = result.decode('utf-8').strip()
            info["gpu_utilization"] = f"{gpu_util}%"
        except:
            info["gpu_utilization"] = "N/A (nvidia-smi not available)"
    
    return info

def create_info_tab():
    """Create the info tab UI elements"""
    with gr.Tab("Info"):
        gr.Markdown("""
        # Sclera Segmentation Application
        
        ## About
        This application provides tools for training, evaluating, and running inference on sclera segmentation models.
        
        ## Author
        
        **Taher Akbari Saeed**  
        Postgraduate Student in Hematology and Blood Transfusion  
        Department of Oncology, Hematology, and Radiotherapy  
        Institute of Postgraduate Education,  
        Pirogov Russian National Research Medical University (RNRMU), Russia
        
        - Email: taherakbarisaeed@gmail.com
        - GitHub: [tayden1990](https://github.com/tayden1990)
        - Telegram: [tayden2023](https://t.me/tayden2023)
        - ORCID: 0000-0002-9517-9773
        
        ## Project Repository
        
        [GitHub: https://github.com/tayden1990/Sclera-Segmentation-with-UI](https://github.com/tayden1990/Sclera-Segmentation-with-UI)
        
        ## Dataset Structure
        
        The application expects your data to be organized as follows:
        
        ```
        sclera_dataset/
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── val/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
        ```
        
        ## Commands
        
        The following commands are available if you want to run scripts directly:
        
        ```bash
        # Training a new model
        python train.py --epochs 20 --batch-size 2 --image-size 256
        
        # Evaluating models
        python evaluate_torch_models.py --checkpoint-dir logs/your_model_dir --test-dir sclera_dataset/test
        
        # Running inference
        python inference_torch.py --weights logs/your_model_dir/checkpoint_019.pth --image your_image.jpg
        ```
        """)
        
        # System Information Section
        gr.Markdown("## System Information")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Available Models")
                model_dirs = "\n".join(find_checkpoints()) if find_checkpoints() else "No models found"
                gr.Textbox(value=model_dirs, label="Model Directories", interactive=False)
            
            with gr.Column():
                gr.Markdown("### Dataset Status")
                
                # Count dataset files
                train_count = 0
                val_count = 0
                test_count = 0
                
                # Try to count files, handling errors if directories don't exist
                try:
                    train_count = len(glob.glob(os.path.join(DATASET_DIR, "train/images/*.jpg")))
                    train_count += len(glob.glob(os.path.join(DATASET_DIR, "train/images/*.png")))
                except:
                    pass
                
                try:
                    val_count = len(glob.glob(os.path.join(DATASET_DIR, "val/images/*.jpg")))
                    val_count += len(glob.glob(os.path.join(DATASET_DIR, "val/images/*.png")))
                except:
                    pass
                
                try:
                    test_count = len(glob.glob(os.path.join(DATASET_DIR, "test/images/*.jpg")))
                    test_count += len(glob.glob(os.path.join(DATASET_DIR, "test/images/*.png")))
                except:
                    pass
                
                dataset_status = f"Training images: {train_count}\n"
                dataset_status += f"Validation images: {val_count}\n"
                dataset_status += f"Test images: {test_count}\n"
                
                gr.Textbox(value=dataset_status, label="Dataset Counts", interactive=False)
        
        # System Resources Section
        gr.Markdown("### System Resources & Environment")
        
        # Display system info in a table format
        sys_info_box = gr.Dataframe(label="System Information", headers=["Parameter", "Value"], interactive=False)
        
        # Function to update system info and format for display
        def update_sys_info():
            info = get_system_info()
            # Format as a list of lists for dataframe display
            return [[k, v] for k, v in info.items()]
        
        # Initial system info
        sys_info_box.value = update_sys_info()
        
        # Refresh button for system info
        refresh_btn = gr.Button("Refresh System Info")
        refresh_btn.click(
            fn=update_sys_info,
            inputs=[],
            outputs=[sys_info_box]
        )

def inference_single(image, model_dir, model_file, threshold=0.5, image_size=512):
    """Run inference on a single image"""
    if image is None:
        return None, "Please upload an image"
        
    if model_dir is None or not model_file:
        return None, "Please select both a model directory and enter a model file"
    
    # Handle cases where model_dir is a list
    if isinstance(model_dir, list):
        model_dir = model_dir[0] if model_dir else None
    
    # Handle case where model_file is a list
    if isinstance(model_file, list):
        model_file = model_file[0] if model_file else None
    
    # Construct full model path
    model_path = os.path.join(MODEL_DIR, model_dir, model_file)
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    
    try:
        # Load model
        model, device = load_model(model_path)
        
        # Preprocess image 
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize while preserving aspect ratio
        original_size = image.size  # PIL returns (width, height)
        ratio = image_size / max(original_size)
        new_size = tuple([int(x * ratio) for x in original_size])
        resized_img = image.resize(new_size, Image.LANCZOS)
        
        # Convert to tensor AND move to the same device as the model
        img_tensor = F.to_tensor(resized_img).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            prediction = model(img_tensor)
            
        # Process prediction
        if len(prediction[0]['masks']) > 0:
            # Get highest scoring mask
            scores = prediction[0]['scores']
            if len(scores) > 0:
                # Get best mask (highest score)
                best_idx = torch.argmax(scores)
                best_score = scores[best_idx].item()
                best_mask = prediction[0]['masks'][best_idx, 0]
                
                # Apply threshold
                binary_mask = best_mask > threshold
                
                # Resize mask back to original size
                mask_np = binary_mask.cpu().numpy().astype(np.uint8) * 255
                mask_pil = Image.fromarray(mask_np)
                mask_pil = mask_pil.resize(original_size, Image.NEAREST)
                
                # Convert image to numpy for processing
                image_np = np.array(image)
                
                # Create a binary mask image (white sclera, black background)
                # Use the same dimensions as the image_np array
                binary_result = np.zeros_like(image_np)
                
                # Convert mask to numpy array with correct dimensions
                mask_np = np.array(mask_pil) > 0
                
                # Make sure dimensions match before indexing
                if mask_np.shape[:2] != binary_result.shape[:2]:
                    # Resize mask to match binary_result dimensions
                    mask_np = np.array(Image.fromarray(mask_np.astype(np.uint8) * 255).resize(
                        (binary_result.shape[1], binary_result.shape[0]), 
                        Image.NEAREST
                    )) > 0
                
                # Apply the mask to create white sclera areas
                binary_result[mask_np] = [255, 255, 255]
                binary_img = Image.fromarray(binary_result)
                
                # Create red overlay
                overlay_color = image_np.copy()
                if len(overlay_color.shape) == 3:  # RGB image
                    overlay_color[mask_np, 0] = 255  # Red channel
                    overlay_color[mask_np, 1] = 0    # Green channel
                    overlay_color[mask_np, 2] = 0    # Blue channel
                else:  # Grayscale image
                    overlay_color = np.stack([image_np] * 3, axis=2)
                    overlay_color[mask_np, 0] = 255  # Red channel
                    overlay_color[mask_np, 1] = 0    # Green channel
                    overlay_color[mask_np, 2] = 0    # Blue channel
                
                # Blend with original
                alpha = 0.5
                result = (1-alpha) * image_np + alpha * overlay_color
                result = result.astype(np.uint8)
                
                # Convert back to PIL
                result_img = Image.fromarray(result)
                
                # Create a combined visualization with both results side by side
                combined = Image.new('RGB', (original_size[0] * 2, original_size[1]))
                combined.paste(result_img, (0, 0))
                combined.paste(binary_img, (original_size[0], 0))
                
                return combined, f"Inference completed. Detection score: {best_score:.4f}"
            else:
                return image, "No detection found with sufficient confidence"
        else:
            return image, "No sclera detected in the image"
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return None, f"Error during inference: {str(e)}\n\nDetails:\n{error_details}"

def run_evaluation(checkpoint_dir, test_dir):
    """Run the evaluation script on the specified checkpoint directory"""
    # Check if directories exist
    if not checkpoint_dir:
        return "Error", "Please select a model directory", None
    
    if not os.path.exists(test_dir):
        return "Error", f"Test directory {test_dir} does not exist", None
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(TEMP_DIR, f"eval_log_{timestamp}.txt")
    output_dir = os.path.join(RESULTS_DIR, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the command
    cmd = [
        "python", "evaluate_torch_models.py",
        "--checkpoint-dir", os.path.join(MODEL_DIR, checkpoint_dir),
        "--test-dir", test_dir,
        "--output-dir", output_dir
    ]
    
    # Run the command and capture output
    with open(output_file, 'w') as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Write output to file and build log string
        log_text = ""
        for line in process.stdout:
            f.write(line)
            f.flush()
            log_text += line
        
        # Wait for process to complete
        process.wait()
    
    # Check if evaluation was successful
    if process.returncode == 0:
        status = f"Evaluation completed successfully. Results in: {output_dir}"
        
        # Try to read and display key results
        try:
            results_csv = os.path.join(output_dir, "model_evaluation_results.csv")
            if os.path.exists(results_csv):
                df = pd.read_csv(results_csv)
                df_sorted = df.sort_values('avg_iou', ascending=False).head(3)
                
                # Create a formatted results string
                results_text = "Top 3 Models by IoU:\n\n"
                for i, row in df_sorted.iterrows():
                    results_text += f"{row['model']}:\n"
                    results_text += f"  IoU: {row['avg_iou']:.4f}\n"
                    results_text += f"  F1 Score: {row['avg_f1']:.4f}\n"
                    results_text += f"  Precision: {row['avg_precision']:.4f}\n"
                    results_text += f"  Recall: {row['avg_recall']:.4f}\n\n"
                
                # Get the visualization image
                vis_image = os.path.join(output_dir, "model_comparison.png")
                if os.path.exists(vis_image):
                    return status, results_text, Image.open(vis_image)
                else:
                    return status, results_text, None
            else:
                return status, "No results CSV found", None
        except Exception as e:
            return status, f"Error reading results: {str(e)}", None
    else:
        status = f"Evaluation failed with error code {process.returncode}. See log for details."
        return status, log_text, None

def create_interface():
    """Create the main Gradio interface"""
    with gr.Blocks(css="footer {text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ddd;}") as demo:
        gr.Markdown("# Sclera Segmentation with UI")
        
        # Create all tabs
        create_training_tab()
        create_inference_tab()
        create_evaluation_tab()
        create_info_tab()
        
        # Add footer with contact info and GitHub link
        gr.Markdown("""
        <footer>
        <p>Developed by Taher Akbari Saeed | <a href="mailto:taherakbarisaeed@gmail.com">taherakbarisaeed@gmail.com</a> | <a href="https://github.com/tayden1990/Sclera-Segmentation-with-UI" target="_blank">GitHub Repository</a></p>
        </footer>
        """)
    
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
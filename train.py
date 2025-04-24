"""
Sclera Segmentation Training Script 
No scikit-image dependency - Everything uses PIL and OpenCV
"""
import os
import torch
import numpy as np
import sys
import time
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import datetime
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2  # Make sure to install opencv-python

# Print debug info
print("Python path:", sys.path)
import torch
print("PyTorch found:", torch.__file__)
print("PyTorch Version:", torch.__version__)
print("TorchVision Version:", torchvision.__version__)

# Set fixed random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ScleraDatasetFixed(Dataset):
    """
    Fixed Sclera dataset without scikit-image dependency.
    Uses PIL and OpenCV for image processing.
    """
    def __init__(self, root, subset="train", image_size=128):
        self.root = root
        self.subset = subset
        self.image_size = image_size
        
        # Get image and mask paths
        self.img_dir = os.path.join(root, subset, "images")
        self.mask_dir = os.path.join(root, subset, "masks")
        
        # Check if structure exists
        if not os.path.exists(self.img_dir):
            # Try alternate structure without "images" subfolder
            self.img_dir = os.path.join(root, subset)
            self.mask_dir = os.path.join(root, "masks")
            
            if not os.path.exists(self.img_dir):
                raise ValueError(f"Could not find images in {os.path.join(root, subset, 'images')} "
                                 f"or {os.path.join(root, subset)}")
        
        # Get list of files
        self.imgs = [f for f in sorted(os.listdir(self.img_dir)) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Found {len(self.imgs)} images in {self.img_dir}")
        
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.imgs[idx])
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            # Try alternative extensions
            for ext in ['.png', '.jpg', '.jpeg']:
                alt_mask_path = os.path.join(self.mask_dir, 
                                            os.path.splitext(self.imgs[idx])[0] + ext)
                if os.path.exists(alt_mask_path):
                    mask_path = alt_mask_path
                    break
        
        if not os.path.exists(mask_path):
            print(f"Warning: No mask found for {self.imgs[idx]}. Creating empty mask.")
            # Create empty mask if not found
            img = Image.open(img_path)
            width, height = img.size
            mask = np.zeros((height, width), dtype=np.uint8)
            img = np.array(img.convert("RGB"))
        else:
            # Load image and mask
            img = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask > 128).astype(np.uint8)  # Convert grayscale to binary
        
        # Resize for consistency - using PIL instead of skimage
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img = np.array(pil_img)
        
        pil_mask = Image.fromarray(mask)
        pil_mask = pil_mask.resize((self.image_size, self.image_size), Image.NEAREST) 
        mask = np.array(pil_mask).astype(np.uint8)
        
        # Get bounding boxes for instances
        pos = np.where(mask)
        if len(pos[0]) > 0:
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes = [[xmin, ymin, xmax, ymax]]
            area = (xmax - xmin) * (ymax - ymin)
            
            # Create target dictionary
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.ones((1,), dtype=torch.int64),
                "masks": torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0),
                "image_id": torch.tensor([idx]),
                "area": torch.tensor([area]),
                "iscrowd": torch.zeros((1,), dtype=torch.int64)
            }
        else:
            # No sclera in image, create empty target
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, self.image_size, self.image_size), dtype=torch.uint8),
                "image_id": torch.tensor([idx]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64)
            }
        
        # Convert image to tensor
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
        return img, target
        
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    """
    Custom collate function for detection data.
    """
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets

def get_model_instance_segmentation(num_classes=2):
    """
    Create Mask R-CNN model with proper architecture for modern TorchVision.
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    # Print model device to debug
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Verify model is on the correct device
    print(f"Model is on device: {next(model.parameters()).device}")
    
    return model

def train_one_epoch(model, optimizer, data_loader, device):
    """
    Train model for one epoch with proper error handling.
    """
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    batch_count = 0
    
    for i, (images, targets) in enumerate(tqdm(data_loader, desc="Training")):
        # Move images and targets to the correct device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        try:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimize
            losses.backward()
            optimizer.step()
            
            # Record loss
            total_loss += losses.item()
            batch_count += 1
            
            # Print batch info
            if i % 5 == 0:  # Print every 5 batches
                print(f"  Batch {i}/{num_batches}: Loss = {losses.item():.4f}")
                if torch.cuda.is_available():
                    print(f"  GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
                
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Return average loss
    return total_loss / max(1, batch_count)

def validate(model, data_loader, device):
    """
    Run validation with error handling.
    """
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    batch_count = 0
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader, desc="Validating")):
            # Move images and targets to the correct device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            try:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Record loss
                total_loss += losses.item()
                batch_count += 1
                
            except Exception as e:
                print(f"Error in validation batch {i}: {e}")
                continue
    
    # Return average loss
    return total_loss / max(1, batch_count)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Sclera Segmentation Training - No scikit-image Version')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--image-size', type=int, default=128, help='image size')
    parser.add_argument('--subset', type=float, default=0.1, help='dataset subset (0-1)')
    parser.add_argument('--log-dir', default='logs/model_default', help='log directory')
    parser.add_argument('--resume', help='path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create dataset
    dataset_train = ScleraDatasetFixed("sclera_dataset", subset="train", image_size=args.image_size)
    dataset_val = ScleraDatasetFixed("sclera_dataset", subset="val", image_size=args.image_size)
    
    # Use subset if requested
    if args.subset < 1.0:
        orig_count = len(dataset_train)
        subset_size = max(1, int(orig_count * args.subset))
        indices = torch.randperm(len(dataset_train)).tolist()[:subset_size]
        dataset_train = torch.utils.data.Subset(dataset_train, indices)
        print(f"Using subset of {len(dataset_train)}/{orig_count} images ({args.subset*100:.1f}%)")
    
    # Create data loaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,  # Try 2 workers, increase if your system can handle it
        pin_memory=True,  # This helps with GPU transfer
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,  # Try 2 workers
        pin_memory=True,  # This helps with GPU transfer
        collate_fn=collate_fn
    )
    
    print(f"Training with {len(dataset_train)} images")
    print(f"Validating with {len(dataset_val)} images")
    
    # Initialize model
    model = get_model_instance_segmentation(num_classes=2)
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {args.resume}")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Check if we're actually using the GPU
        if torch.cuda.is_available():
            print(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        # Train and validate
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        
        if torch.cuda.is_available():
            print(f"GPU Memory after training: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        
        val_loss = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.log_dir, f"checkpoint_{epoch:03d}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.log_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return 0

if __name__ == "__main__":
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
        # Monitor GPU memory
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    else:
        print("WARNING: CUDA is not available, using CPU")
    sys.exit(main())
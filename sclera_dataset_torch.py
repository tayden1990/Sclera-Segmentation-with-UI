import os
import torch
import torch.utils.data
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F

class ScleraDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, subset="train"):
        self.root = root
        self.transforms = transforms
        self.subset = subset
        
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
        else:
            # Load image and mask
            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask)
            mask = (mask > 128).astype(np.uint8)  # Convert grayscale to binary
        
        # Convert image to numpy array if not already
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        # Get bounding boxes for instances
        # In our case, we have only one instance (sclera)
        pos = np.where(mask)
        if len(pos[0]) > 0:
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes = [[xmin, ymin, xmax, ymax]]
            area = (xmax - xmin) * (ymax - ymin)
            
            # Create target dictionary
            target = {}
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.ones((1,), dtype=torch.int64)  # class 1 = sclera
            target["masks"] = torch.as_tensor(mask, dtype=torch.uint8).unsqueeze(0)
            target["image_id"] = torch.tensor([idx])
            target["area"] = torch.tensor([area])
            target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)
        else:
            # No sclera in image, create empty target
            height, width = mask.shape
            target = {}
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)
            target["image_id"] = torch.tensor([idx])
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
        
        # Convert image to tensor
        img = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
        
    def __len__(self):
        return len(self.imgs)
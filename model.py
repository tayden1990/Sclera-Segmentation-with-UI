import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class ScleraSegmentationModel:
    """Wrapper for Mask R-CNN model for sclera segmentation"""
    
    def __init__(self, num_classes=2, pretrained=True):
        """Initialize model with specified number of classes"""
        # Initialize with pretrained weights if requested
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        
        # Replace the box predictor
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Replace the mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    def to(self, device):
        """Move model to specified device"""
        self.model.to(device)
        return self
    
    def train(self):
        """Set model to training mode"""
        self.model.train()
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        return self
    
    def load_state_dict(self, state_dict):
        """Load state dictionary into model"""
        self.model.load_state_dict(state_dict)
    
    def forward(self, images, targets=None):
        """Forward pass through the model"""
        return self.model(images, targets)
    
    def __call__(self, *args, **kwargs):
        """Direct call to model's forward method"""
        return self.model(*args, **kwargs)
    
    def state_dict(self):
        """Get state dictionary from model"""
        return self.model.state_dict()
    
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()

def create_model(num_classes=2, pretrained=True, weights_path=None, device=None):
    """Factory function to create and initialize a model"""
    model = ScleraSegmentationModel(num_classes, pretrained)
    
    # Load weights if provided
    if weights_path:
        checkpoint = torch.load(weights_path, map_location='cpu' if device is None else device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    # Move to device if provided
    if device:
        model.to(device)
    
    return model
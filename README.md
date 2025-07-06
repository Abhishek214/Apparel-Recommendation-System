# download_efficientdet_torchvision.py
import torch
import torchvision.models as models

# Try torchvision's EfficientDet
try:
    model = models.efficientnet_b1(pretrained=True)
    
    # Modify for object detection (simple approach)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 4)  # 4 classes
    
    # Save complete model
    torch.save(model, 'efficientdet_alternative.pth')
    print("✅ Alternative model created!")
    
except Exception as e:
    print(f"Error: {e}")

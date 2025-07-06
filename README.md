download_complete_model.py
import torch
import timm

# Create the EfficientDet-D1 architecture
model = timm.create_model('tf_efficientdet_d1', pretrained=False, num_classes=80)  # COCO classes

# Load the downloaded weights
state_dict = torch.load('tf_efficientdet_d1-4c7ebaf2.pth', map_location='cpu')
model.load_state_dict(state_dict)

# Save the complete model
torch.save(model, 'efficientdet_d1_complete_model.pth')
print("✅ Complete model saved!")

# Check file size
import os
size_mb = os.path.getsize('efficientdet_d1_complete_model.pth') / (1024*1024)
print(f"Complete model size: {size_mb:.1f} MB")

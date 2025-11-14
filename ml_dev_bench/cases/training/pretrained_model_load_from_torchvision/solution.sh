#!/bin/bash
set -e

# Ensure Poetry is in PATH
export PATH="/opt/poetry/bin:$PATH"

# Create a Python script to download and save ResNet18 model
cat > load_resnet18.py << 'EOF'
import json
from pathlib import Path
import torch
import torchvision.models as models

# Download and load the pretrained ResNet18 model
model_name = "resnet18"
print(f"Downloading {model_name} from torchvision...")

model = models.resnet18(pretrained=True)

# Save model to disk
model_path = Path("./resnet18_model.pth")
print(f"Saving model to {model_path}...")
torch.save(model.state_dict(), model_path)

# Count the number of parameters
num_parameters = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_parameters:,}")

# ResNet18 input shape is [3, 224, 224] (Channels, Height, Width)
input_shape = [3, 224, 224]

# Create model_info.json with all required fields
model_info = {
    "model_name": model_name,
    "model_path": str(model_path),
    "architecture": {
        "input_shape": input_shape,
        "num_parameters": num_parameters
    }
}

# Save model_info.json to workspace root
info_path = Path("./model_info.json")
with open(info_path, "w") as f:
    json.dump(model_info, f, indent=2)

print(f"\nModel info saved to {info_path}")
print(f"Model architecture:")
print(f"  - input_shape: {input_shape}")
print(f"  - num_parameters: {num_parameters:,}")
print("Done!")
EOF

# Run the Python script using poetry
cd /app && poetry run python load_resnet18.py

echo "✅ ResNet18 model downloaded and saved successfully"

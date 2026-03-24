import torch
import torch.nn as nn
from torchvision import models


def create_efficientnet_b3_model(num_classes):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model

if __name__ == '__main__':
    num_classes = 3
    model_path = "best_model_128.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_efficientnet_b3_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, 128, 128, device=device)

    onnx_path = "model_128.onnx"
    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=['input'], output_names=['output'],
        opset_version=13
    )

    print(f"ONNX model saved to {onnx_path}")

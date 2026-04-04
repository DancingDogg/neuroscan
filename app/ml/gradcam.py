import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_last_conv_layer(model, model_name):
    if "resnet" in model_name:
        return model.layer4[-1].conv3 if hasattr(model.layer4[-1], "conv3") else model.layer4[-1].conv2
    elif "densenet" in model_name:
        return model.features[-1]
    elif "efficientnet" in model_name:
        return model.conv_head
    else:
        raise ValueError(f"No GradCAM layer defined for {model_name}")

class GradCAM:
    def __init__(self, model, model_name):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        target_layer = get_last_conv_layer(model, model_name)

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(device)
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        self.model.zero_grad()
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

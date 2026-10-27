import torch
import torchvision

model=torchvision.models.mobilenet_v2(weights=None)
model.eval()
dummy_input=torch.randn(1,3,224,224) #1 Image, 3 colours, 224x224 pixels

torch.onnx.export(
    model,
    dummy_input,
    "mobilenet_v2.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
)
print("Saved mobilenet_v2.onnx")
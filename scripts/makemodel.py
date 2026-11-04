from pathlib import Path

import torch
import torchvision

weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
model = torchvision.models.mobilenet_v2(weights=weights)
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)  # 1 image, 3 colours, 224x224 pixels

out_path = Path(__file__).resolve().parent.parent / "models" / "mobilenet_v2.onnx"
out_path.parent.mkdir(exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    str(out_path),
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
)
print(f"Saved {out_path}")
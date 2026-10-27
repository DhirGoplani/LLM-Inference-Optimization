#sanity check
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("mobilenet_v2.onnx")
inp = session.get_inputs()[0]
print(inp.name, inp.shape, inp.type)

np.random.seed(0)
data = np.random.rand(1, 3, 224, 224).astype(np.float32)
output = session.run(None, {inp.name: data})
print(output[0].shape)
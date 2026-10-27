import sys
sys.path.append("..")

from inferbench.common.benchmarktypes import BenchmarkConfig, BenchmarkResult

config = BenchmarkConfig(model_path="../models/mobilenet_v2.onnx", num_threads=4)
result = BenchmarkResult(config=config, success=True, correct=True, latency_mean_ms=12.5)

print(result.is_valid)
print(result.to_dict())
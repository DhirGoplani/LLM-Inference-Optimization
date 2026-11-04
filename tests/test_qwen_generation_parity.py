"""Optional numerical parity test against the original Hugging Face model.

Run explicitly with:
    RUN_QWEN_PARITY=1 python -m pytest -q tests/test_qwen_generation_parity.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen2.5_0.5b"
ONNX_PATH = MODEL_DIR / "model.onnx"
RUN_PARITY = os.environ.get("RUN_QWEN_PARITY") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_PARITY,
    reason="Set RUN_QWEN_PARITY=1 to run the optional Hugging Face parity test",
)


def _hf_greedy_tokens(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    import torch

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    generated: list[int] = []
    past = None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(
                input_ids=input_ids if past is None else input_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break
            input_ids = torch.tensor([[token]], dtype=input_ids.dtype)
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype)],
                dim=1,
            )
            past = outputs.past_key_values
    return generated


@pytest.mark.parametrize(
    "prompt",
    [
        "The capital of France is",
        "Explain why caching improves decoder inference.",
        "The sky is",
    ],
)
def test_onnx_greedy_tokens_match_huggingface(prompt):
    if not ONNX_PATH.is_file():
        pytest.skip(f"ONNX model not found: {ONNX_PATH}")

    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from models.generation import GenerationConfig, GenerationRequest, OnnxTextGenerator

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        local_files_only=False,
    ).eval()
    onnx_result = OnnxTextGenerator(
        MODEL_DIR, tokenizer=tokenizer
    ).generate(
        GenerationRequest(
            prompt,
            GenerationConfig(max_new_tokens=20, seed=0),
        )
    )
    hf_tokens = _hf_greedy_tokens(hf_model, tokenizer, prompt, 20)
    np.testing.assert_array_equal(
        np.asarray(onnx_result.generated_token_ids[:20]),
        np.asarray(hf_tokens[:20]),
        err_msg=f"ONNX and Hugging Face greedy outputs diverged for prompt {prompt!r}",
    )

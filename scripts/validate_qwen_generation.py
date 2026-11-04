#!/usr/bin/env python3
"""Compare Qwen ONNX greedy decoding with the original Hugging Face model."""

from __future__ import annotations

import argparse
from pathlib import Path

from models.generation import GenerationConfig, GenerationRequest, OnnxTextGenerator
from models.tokenizer import load_tokenizer

PROMPTS = (
    "The capital of France is",
    "Explain why caching improves decoder inference.",
    "The sky is",
)


def hf_greedy_tokens(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/qwen2.5_0.5b")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    model_dir = Path(args.model_dir).expanduser().resolve()
    tokenizer = load_tokenizer(model_dir)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.float16
    ).eval()
    generator = OnnxTextGenerator(model_dir, tokenizer=tokenizer)
    passed = True
    for prompt in PROMPTS:
        onnx_tokens = list(generator.generate(GenerationRequest(
            prompt, GenerationConfig(max_new_tokens=args.max_new_tokens)
        )).generated_token_ids[:20])
        hf_tokens = hf_greedy_tokens(hf_model, tokenizer, prompt, args.max_new_tokens)
        match = onnx_tokens == hf_tokens[:20]
        passed = passed and match
        print(f"{'PASS' if match else 'FAIL'}: {prompt!r}")
        print(f"  ONNX: {onnx_tokens}")
        print(f"  HF:   {hf_tokens[:20]}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

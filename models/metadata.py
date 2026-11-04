from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any


def calculate_approx_parameter_count(cfg: dict[str, Any]) -> int | None:
    """Derive approximate parameter count from causal LM architecture configuration.

    Components Included:
    - Token embedding matrix (vocab_size * hidden_size).
    - Multi-Head / Grouped-Query Attention linear projections per layer:
      - Q projection: hidden_size * (num_heads * head_dim)
      - K projection: hidden_size * (num_kv_heads * head_dim)
      - V projection: hidden_size * (num_kv_heads * head_dim)
      - O projection: (num_heads * head_dim) * hidden_size
    - Attention projection biases per layer:
      - Q, K, V projection biases where present (e.g. Qwen2 architectures:
        (num_heads + 2 * num_kv_heads) * head_dim).
    - Feed-Forward / MLP projections per layer:
      - SwiGLU / Gated MLP architectures (e.g. Qwen, Llama, Gemma, Mistral):
        gate_proj + up_proj + down_proj = 3 * hidden_size * intermediate_size
      - Standard 2-projection MLPs (e.g. GPT-2, Falcon):
        up_proj + down_proj = 2 * hidden_size * intermediate_size
      - MLP biases when explicitly configured (mlp_bias=True).
    - Normalization layers:
      - Pre-attention (input) and post-attention RMSNorms/LayerNorms per layer:
        2 * hidden_size per layer.
      - Final output normalization: 1 * hidden_size.
    - Language model output head:
      - If untied (tie_word_embeddings=False): vocab_size * hidden_size.
      - If tied (tie_word_embeddings=True): 0 additional parameters.

    Components Excluded:
    - Rotary Position Embeddings (RoPE / sinusoidal): computed dynamically at runtime
      without learned parameter weights.
    - Runtime inference buffers and KV-cache states (activations, not model weights).
    - Multi-modal components or vision encoders (causal LM text backbone only).

    Validation & Fallback:
    - Validates that hidden_size, vocab_size, num_hidden_layers, intermediate_size,
      and num_attention_heads are strictly positive integers.
    - Validates that hidden_size is divisible by num_attention_heads (or explicit head_dim).
    - Validates that num_attention_heads is greater than or equal to num_key_value_heads.
    - Returns None if configuration is missing, malformed, or insufficient to compute
      a reliable estimate.
    """
    if not isinstance(cfg, dict):
        return None

    hidden_size = cfg.get("hidden_size")
    vocab_size = cfg.get("vocab_size")
    num_layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    intermediate_size = cfg.get("intermediate_size")
    num_heads = cfg.get("num_attention_heads") or cfg.get("num_heads")

    for val in (hidden_size, vocab_size, num_layers, intermediate_size, num_heads):
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            return None

    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    if isinstance(num_kv_heads, bool) or not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        return None

    if num_kv_heads > num_heads or num_heads % num_kv_heads != 0:
        return None

    head_dim = cfg.get("head_dim")
    if head_dim is not None:
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            return None
    else:
        if hidden_size % num_heads != 0:
            return None
        head_dim = hidden_size // num_heads

    # 1. Embeddings
    embed_params = vocab_size * hidden_size

    # 2. Attention projections per layer
    q_proj = hidden_size * (num_heads * head_dim)
    k_proj = hidden_size * (num_kv_heads * head_dim)
    v_proj = hidden_size * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * hidden_size
    attn_params = q_proj + k_proj + v_proj + o_proj

    model_type = str(cfg.get("model_type", "")).lower()
    has_qkv_bias = False
    if "qwen" in model_type:
        has_qkv_bias = cfg.get("attention_bias", True)
    elif cfg.get("attention_bias", False) or cfg.get("qkv_bias", False):
        has_qkv_bias = True

    if has_qkv_bias:
        attn_params += (num_heads * head_dim) + 2 * (num_kv_heads * head_dim)

    # 3. MLP projections per layer
    hidden_act = str(cfg.get("hidden_act", "")).lower()
    is_swiglu = (
        "silu" in hidden_act
        or "swiglu" in hidden_act
        or any(arch in model_type for arch in ("qwen", "llama", "gemma", "mistral", "mixtral", "phi"))
        or bool(cfg.get("gated_mlp", False))
    )
    mlp_mult = 3 if is_swiglu else 2
    mlp_params = mlp_mult * hidden_size * intermediate_size

    if cfg.get("mlp_bias", False):
        mlp_params += (2 if is_swiglu else 1) * intermediate_size + hidden_size

    # 4. Normalization layers per layer
    norm_params = 2 * hidden_size

    total_layers = (attn_params + mlp_params + norm_params) * num_layers
    final_norm = hidden_size

    # 5. Language model head
    is_tied = cfg.get("tie_word_embeddings", False)
    lm_head = 0 if is_tied else (vocab_size * hidden_size)

    return embed_params + total_layers + final_norm + lm_head


def resolve_canonical_model_id(cfg: dict[str, Any], dir_path: Path) -> str:
    """Resolve canonical Hugging Face model ID from configuration or architecture signature."""
    raw_id = cfg.get("_name_or_path")
    if isinstance(raw_id, str) and raw_id.strip():
        s = raw_id.strip()
        is_local_path = (
            s.startswith(("/", "./", "../", "~/", "models/"))
            or "\\" in s
            or Path(s).is_dir()
            or Path(s).is_file()
        )
        if not is_local_path:
            return s

    model_type = str(cfg.get("model_type", "")).lower()
    layers = cfg.get("num_hidden_layers") or cfg.get("num_layers")
    hidden = cfg.get("hidden_size")
    intermediate = cfg.get("intermediate_size")
    vocab = cfg.get("vocab_size")

    # Qwen/Qwen2.5-0.5B-Instruct
    if (
        model_type == "qwen2"
        and layers == 24
        and hidden == 896
        and intermediate == 4864
        and vocab == 151936
    ):
        return "Qwen/Qwen2.5-0.5B-Instruct"

    # Qwen/Qwen2.5-1.5B-Instruct
    if (
        model_type == "qwen2"
        and layers == 28
        and hidden == 1536
        and intermediate == 8960
        and vocab == 151936
    ):
        return "Qwen/Qwen2.5-1.5B-Instruct"

    # meta-llama/Llama-3.2-1B-Instruct
    if (
        model_type == "llama"
        and layers == 16
        and hidden == 2048
        and intermediate == 8192
        and vocab == 128256
    ):
        return "meta-llama/Llama-3.2-1B-Instruct"

    # google/gemma-2-2b-it
    if (
        model_type == "gemma2"
        and layers == 26
        and hidden == 2304
        and intermediate == 9216
        and vocab == 256000
    ):
        return "google/gemma-2-2b-it"

    return raw_id or dir_path.name


def detect_onnx_precision(model_dir: Path) -> tuple[str, str]:
    """Inspect ONNX model files in directory to determine weight precision."""
    model_dir = Path(model_dir)
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        return "unknown", "no_onnx_file_found"

    primary_model = model_dir / "model.onnx"
    if not primary_model.is_file():
        primary_model = onnx_files[0]

    try:
        import onnx
        model = onnx.load(str(primary_model), load_external_data=False)
        float_types = set()
        for init in model.graph.initializer:
            if init.data_type in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT, onnx.TensorProto.BFLOAT16):
                float_types.add(init.data_type)

        if not float_types:
            return "unknown", "onnx_graph (no float initializers found)"

        if float_types == {onnx.TensorProto.FLOAT16}:
            return "fp16", "onnx_graph (consistent float16 initializers)"
        elif float_types == {onnx.TensorProto.FLOAT}:
            return "fp32", "onnx_graph (consistent float32 initializers)"
        else:
            return "mixed", "onnx_graph (mixed precision initializers)"
    except Exception as exc:
        warnings.warn(
            f"Failed to inspect precision from {primary_model}: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return "unknown", f"inspection_error ({type(exc).__name__})"


def load_llm_metadata(model_dir: str | Path) -> dict[str, Any]:
    """Parse configuration files in a model directory and build structured LLM metadata."""
    dir_path = Path(model_dir).expanduser().resolve()
    config_path = dir_path / "config.json"

    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found in model directory: {dir_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # Optional generation_config.json with explicit warning on parse error
    gen_cfg: dict[str, Any] = {}
    gen_config_path = dir_path / "generation_config.json"
    if gen_config_path.is_file():
        try:
            with open(gen_config_path, encoding="utf-8") as f:
                gen_cfg = json.load(f)
        except Exception as exc:
            warnings.warn(
                f"Could not parse generation_config.json at {gen_config_path}: {exc}",
                UserWarning,
                stacklevel=2,
            )

    hidden_size = cfg.get("hidden_size")
    num_heads = cfg.get("num_attention_heads")
    num_kv_heads = cfg.get("num_key_value_heads", num_heads)
    head_dim = cfg.get("head_dim") or ((hidden_size // num_heads) if (hidden_size and num_heads) else None)

    approx_params = calculate_approx_parameter_count(cfg)
    canonical_id = resolve_canonical_model_id(cfg, dir_path)
    precision, precision_source = detect_onnx_precision(dir_path)

    # Calculate model size on disk
    onnx_files = sorted(dir_path.glob("*.onnx"))
    model_size_mb = None
    if onnx_files:
        total_size = sum(p.stat().st_size for p in onnx_files)
        for onnx_file in onnx_files:
            data_file = onnx_file.parent / f"{onnx_file.name}.data"
            if data_file.is_file():
                total_size += data_file.stat().st_size
            data_file_alt = onnx_file.parent / f"{onnx_file.name}_data"
            if data_file_alt.is_file():
                total_size += data_file_alt.stat().st_size
        model_size_mb = round(total_size / (1024 * 1024), 2)

    return {
        "model_id": canonical_id,
        "model_type": cfg.get("model_type", "unknown"),
        "architecture": (cfg.get("architectures") or ["Unknown"])[0],
        "parameter_count_approx": approx_params,
        "parameter_count_basis": (
            "configuration-derived (approximate: embeds, attention with biases, SwiGLU MLP, norms)"
            if approx_params is not None
            else "unknown (insufficient configuration)"
        ),
        "precision": precision,
        "precision_source": precision_source,
        "model_dir": str(dir_path),
        "tokenizer_dir": str(dir_path) if (dir_path / "tokenizer.json").is_file() else None,
        "hidden_size": hidden_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "intermediate_size": cfg.get("intermediate_size"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings", False),
        "bos_token_id": cfg.get("bos_token_id", gen_cfg.get("bos_token_id")),
        "eos_token_id": cfg.get("eos_token_id", gen_cfg.get("eos_token_id")),
        "pad_token_id": cfg.get("pad_token_id", gen_cfg.get("pad_token_id")),
        "model_size_mb": model_size_mb,
    }


def get_qwen2_5_0_5b_metadata(model_dir: str | Path | None = None) -> dict[str, Any]:
    """Retrieve verified metadata for Qwen2.5-0.5B-Instruct."""
    if model_dir is not None:
        p = Path(model_dir)
        if (p / "config.json").is_file():
            return load_llm_metadata(p)
    else:
        default_dir = Path("models/qwen2.5_0.5b")
        if default_dir.is_dir() and (default_dir / "config.json").is_file():
            return load_llm_metadata(default_dir)

    # Static reference fallback when local model directory is unavailable
    return {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_type": "qwen2",
        "architecture": "Qwen2ForCausalLM",
        "parameter_count_approx": 494_032_768,
        "parameter_count_basis": "static_reference (verified Qwen2.5-0.5B-Instruct)",
        "precision": "fp16",
        "precision_source": "static_reference",
        "model_dir": "models/qwen2.5_0.5b",
        "tokenizer_dir": "models/qwen2.5_0.5b",
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "num_hidden_layers": 24,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "intermediate_size": 4864,
        "tie_word_embeddings": True,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "pad_token_id": 151643,
    }

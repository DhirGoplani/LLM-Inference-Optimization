"""Streamlit dashboard for InferBench AutoTune."""

import asyncio
import json
from pathlib import Path
import time
import streamlit as st

from inferbench.dashboard.api_client import (
    DEFAULT_API_URL,
    InferBenchApiClient,
    compute_latency_speedup,
    compute_throughput_change,
    discover_models,
    format_latency_speedup,
    format_throughput_change,
)
from models.inspection import inspect_model_configuration
from models.generation import (
    GenerationConfig,
    GenerationRequest,
    OnnxTextGenerator,
)
from models.batching import BatchConfig, DynamicBatcher
from models.provider_diagnostics import diagnose_execution_providers

# Page configuration
st.set_page_config(
    page_title="InferBench AutoTune",
    page_icon=None,
    layout="wide",
)

st.title("InferBench AutoTune")
st.caption("Benchmarking and Optimization Dashboard for ONNX Model Inference")

# Sidebar: Backend configuration and health check
st.sidebar.header("Backend Connection")
backend_url = st.sidebar.text_input(
    "API URL",
    value=DEFAULT_API_URL,
    help="URL of the running InferBench AutoTune FastAPI server",
)

client = InferBenchApiClient(base_url=backend_url)
health = client.check_health()

if health.ok:
    st.sidebar.success(f"**Connected**\n\nService: `{health.service}`\nStatus: `{health.status}`")
else:
    st.sidebar.error(f"**Offline / Unreachable**\n\n{health.error}")
    st.warning(
        f"**FastAPI backend is unreachable at `{backend_url}`**.\n\n"
        "Please start the API backend from the project root:\n"
        "```bash\n"
        ".venv/bin/uvicorn inferbench.api.main:app --host 127.0.0.1 --port 8000\n"
        "```\n"
        f"**Error Details**: {health.error}"
    )

# Model Selection Section
st.subheader("1. Model Selection")
discovered_models = discover_models("models")

if not discovered_models:
    st.info(
        "No `.onnx` models found in the `models/` directory. "
        "You can specify a path manually below (relative to the repository root or absolute)."
    )
    model_path = st.text_input(
        "Model Path",
        value="",
        placeholder="e.g. models/mobilenet_v2.onnx or /absolute/path/to/model.onnx",
        help="Path to the ONNX model file (relative or absolute)",
    )
else:
    model_choice_mode = st.radio(
        "Model Selection Mode",
        ["Choose from discovered models", "Enter path manually"],
        horizontal=True,
    )
    if model_choice_mode == "Choose from discovered models":
        model_path = st.selectbox(
            "Discovered Models",
            options=discovered_models,
            help="Models discovered in the models/ directory",
        )
    else:
        model_path = st.text_input(
            "Model Path",
            value="",
            placeholder="e.g. models/mobilenet_v2.onnx or /path/to/model.onnx",
            help="Path to the ONNX model file (relative or absolute)",
        )

# Model Inspection & Diagnostics Section
with st.expander("Model Inspection & Diagnostics", expanded=False):
    if not model_path or not model_path.strip():
        st.info("Select or enter a valid ONNX model path above to inspect model architecture and diagnostics.")
    else:
        col_inspect, col_diag = st.columns([1, 1])
        with col_inspect:
            inspect_clicked = st.button("Inspect Model", key="btn_inspect_model")
        with col_diag:
            diag_clicked = st.button("Run Provider Diagnostics", key="btn_run_diagnostics")

        if inspect_clicked:
            with st.spinner("Inspecting ONNX model graph and runtime configuration..."):
                try:
                    insp_data = inspect_model_configuration(model_path)
                    st.session_state["inspection_model_path"] = model_path
                    st.session_state["inspection_data"] = insp_data
                    st.session_state["inspection_error"] = None
                except Exception as exc:
                    st.session_state["inspection_model_path"] = model_path
                    st.session_state["inspection_data"] = None
                    st.session_state["inspection_error"] = f"{type(exc).__name__}: {exc}"

        if diag_clicked:
            with st.spinner("Running execution provider diagnostics..."):
                try:
                    diag_data = diagnose_execution_providers(model_path)
                    st.session_state["diagnostics_model_path"] = model_path
                    st.session_state["diagnostics_data"] = diag_data
                    st.session_state["diagnostics_error"] = None
                except Exception as exc:
                    st.session_state["diagnostics_model_path"] = model_path
                    st.session_state["diagnostics_data"] = None
                    st.session_state["diagnostics_error"] = f"{type(exc).__name__}: {exc}"

        # Display Inspection Results if available for current model path
        if st.session_state.get("inspection_model_path") == model_path:
            if st.session_state.get("inspection_error"):
                st.error(f"Inspection Failed: {st.session_state['inspection_error']}")
            elif st.session_state.get("inspection_data"):
                insp = st.session_state["inspection_data"]
                st.markdown("#### Model Specification & Architecture")

                col_stat, col_ops, col_count = st.columns(3)
                with col_stat:
                    status_text = (
                        "Session Created"
                        if insp.get("runtime_status") == "session_created"
                        else "Session Creation Failed"
                    )
                    st.metric("Runtime Status", status_text)
                with col_ops:
                    opsets = insp.get("opsets", [])
                    opset_str = (
                        ", ".join(
                            f"{op.get('domain', 'ai.onnx')} v{op.get('version', '')}"
                            for op in opsets
                        )
                        if opsets
                        else "N/A"
                    )
                    st.metric("ONNX Opset", opset_str)
                with col_count:
                    op_count = len(insp.get("operators", []))
                    st.metric("Unique Operators", f"{op_count}")

                warnings = insp.get("warnings", [])
                if warnings:
                    st.markdown("**Model Warnings**")
                    for w in warnings:
                        st.warning(w)

                st.markdown("**Input Signatures**")
                inputs = insp.get("inputs", [])
                if inputs:
                    input_rows = [
                        {
                            "Name": item.get("name", ""),
                            "Shape": str(item.get("shape", [])),
                            "Data Type": item.get("dtype", ""),
                        }
                        for item in inputs
                    ]
                    st.dataframe(input_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No graph inputs detected.")

                st.markdown("**Output Signatures**")
                outputs = insp.get("outputs", [])
                if outputs:
                    output_rows = [
                        {
                            "Name": item.get("name", ""),
                            "Shape": str(item.get("shape", [])),
                            "Data Type": item.get("dtype", ""),
                        }
                        for item in outputs
                    ]
                    st.dataframe(output_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No graph outputs detected.")

                operators = insp.get("operators", [])
                if operators:
                    with st.expander(f"View All {len(operators)} Unique Operators", expanded=False):
                        st.write(", ".join(f"`{op}`" for op in operators))

        # Display Provider Diagnostics if available for current model path
        if st.session_state.get("diagnostics_model_path") == model_path:
            if st.session_state.get("diagnostics_error"):
                st.error(f"Diagnostics Failed: {st.session_state['diagnostics_error']}")
            elif st.session_state.get("diagnostics_data"):
                diag = st.session_state["diagnostics_data"]
                st.markdown("---")
                st.markdown("#### Execution Provider Diagnostics")

                diag_results = diag.get("results", [])
                if diag_results:
                    diag_rows = []
                    for r in diag_results:
                        observed = (
                            ", ".join(
                                f"{k}: {v}"
                                for k, v in r.get("observed_node_providers", {}).items()
                            )
                            or "None"
                        )
                        sess_provs = ", ".join(r.get("session_providers", [])) or "None"
                        diag_rows.append({
                            "Provider": r.get("provider", ""),
                            "Session Created": "Yes" if r.get("session_created") else "No",
                            "Inference Succeeded": "Yes" if r.get("inference_succeeded") else "No",
                            "Session Providers": sess_provs,
                            "Observed Node Providers": observed,
                            "Error": r.get("error") or "None",
                        })
                    st.dataframe(diag_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No diagnostic results available.")

                note = diag.get("note", "")
                if note:
                    st.info(
                        f"**Diagnostic Note**:\n\n{note}\n\n"
                        f"*Zero-valued sample inputs were used for this diagnostic test inference. "
                        f"Observed provider assignments apply only to this test run.*"
                    )

# Mode Selector
workflow_mode = st.radio(
    "Workflow Mode",
    ["LLM Generation", "LLM Batching", "AutoTune Optimization", "Single Benchmark"],
    horizontal=True,
    help=(
        "Choose local LLM generation, automated multi-candidate tuning, "
        "or a single manual benchmark"
    ),
)

def fmt_ms(val):
    return f"{val:.2f} ms" if isinstance(val, (int, float)) else "N/A"

def fmt_float(val, unit=""):
    return f"{val:.1f} {unit}".strip() if isinstance(val, (int, float)) else "N/A"


# =========================================================================
# Mode 0: LLM Generation
# =========================================================================
if workflow_mode == "LLM Generation":
    st.subheader("2. LLM Generation")
    st.info(
        "Run deterministic greedy generation directly against a local decoder-only "
        "ONNX model and tokenizer. Cached decoding reuses the model's KV-cache "
        "when the export exposes compatible past and present tensors."
    )

    prompt = st.text_area(
        "Prompt",
        value="Explain KV cache in one sentence.",
        height=120,
        help="Text passed to the local tokenizer before the ONNX prefill pass.",
    )
    gen_col1, gen_col2, gen_col3 = st.columns(3)
    with gen_col1:
        max_new_tokens = st.number_input(
            "Maximum New Tokens",
            min_value=0,
            value=32,
            step=1,
        )
    with gen_col2:
        use_cache = st.checkbox(
            "Use KV cache",
            value=True,
            help="Disable this only for exports without compatible cache tensors.",
        )
    with gen_col3:
        generation_seed = st.number_input(
            "Seed",
            min_value=0,
            value=0,
            step=1,
        )

    generate_btn = st.button(
        "Generate Text",
        type="primary",
        use_container_width=True,
    )

    if generate_btn:
        cleaned_model_path = model_path.strip() if model_path else ""
        if not cleaned_model_path:
            st.error("Select an ONNX model before generating text.")
        elif not prompt.strip():
            st.error("Enter a non-empty prompt before generating text.")
        else:
            with st.spinner("Loading tokenizer and running ONNX generation..."):
                try:
                    generator = OnnxTextGenerator(
                        Path(cleaned_model_path).parent,
                        model_path=cleaned_model_path,
                    )
                    result = generator.generate(
                        GenerationRequest(
                            prompt=prompt,
                            config=GenerationConfig(
                                max_new_tokens=int(max_new_tokens),
                                use_cache=use_cache,
                                seed=int(generation_seed),
                            ),
                        )
                    )
                except Exception as exc:
                    st.error(f"Generation failed: {type(exc).__name__}: {exc}")
                else:
                    st.markdown("### Generated Text")
                    st.write(result.text)

                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    with metric_col1:
                        st.metric("Prompt Tokens", result.prompt_token_count)
                    with metric_col2:
                        st.metric("Generated Tokens", result.generated_token_count)
                    with metric_col3:
                        st.metric("TTFT", fmt_ms(result.time_to_first_token_ms))
                    with metric_col4:
                        st.metric(
                            "Token Throughput",
                            fmt_float(result.tokens_per_second, "tokens/s"),
                        )

                    with st.expander("Generation Details"):
                        detail_col1, detail_col2, detail_col3 = st.columns(3)
                        with detail_col1:
                            st.write(f"**KV cache used:** `{result.cache_used}`")
                            st.write(f"**Prefill latency:** {fmt_ms(result.prefill_latency_ms)}")
                        with detail_col2:
                            st.write(f"**Decode latency:** {fmt_ms(result.decode_latency_ms)}")
                            st.write(
                                f"**Inter-token latency:** "
                                f"{fmt_ms(result.inter_token_latency_ms)}"
                            )
                        with detail_col3:
                            st.write(f"**Model family:** `{generator.model_metadata.model_family or 'N/A'}`")
                            st.write(
                                f"**Cache layers:** "
                                f"`{generator.model_metadata.cache.layer_count}`"
                            )


# =========================================================================
# Mode 1: LLM Batching
# =========================================================================
elif workflow_mode == "LLM Batching":
    st.subheader("2. Dynamic LLM Batching")
    st.info(
        "Submit multiple prompts together. Requests are grouped by prompt-length "
        "bucket, padded only within each group, and returned in the original order."
    )

    batch_prompts = st.text_area(
        "Prompts (one per line)",
        value=(
            "Explain KV cache in one sentence.\n"
            "What is dynamic batching?\n"
            "Why are prompt-length buckets useful?"
        ),
        height=150,
        help="Each non-empty line becomes one generation request.",
    )
    batch_col1, batch_col2, batch_col3 = st.columns(3)
    with batch_col1:
        batch_max_size = st.number_input(
            "Maximum Batch Size", min_value=1, value=4, step=1
        )
        batch_total_tokens = st.number_input(
            "Maximum Total Batch Tokens", min_value=1, value=8192, step=1
        )
    with batch_col2:
        batch_window_ms = st.number_input(
            "Batching Window (ms)", min_value=0.0, value=5.0, step=1.0
        )
        batch_padding_ratio = st.number_input(
            "Maximum Padding Ratio",
            min_value=0.0,
            value=0.5,
            step=0.1,
            format="%.2f",
        )
    with batch_col3:
        batch_bucket_text = st.text_input(
            "Prompt Bucket Boundaries",
            value="128,512,2048,4096",
            help="Comma-separated maximum token lengths; leave blank to disable buckets.",
        )
        batch_max_new_tokens = st.number_input(
            "Maximum New Tokens", min_value=0, value=32, step=1
        )

    batch_cache = st.checkbox("Use KV cache", value=True, key="batch_use_cache")
    batch_btn = st.button(
        "Run Batched Generation",
        type="primary",
        use_container_width=True,
    )

    if batch_btn:
        cleaned_model_path = model_path.strip() if model_path else ""
        prompts = [line.strip() for line in batch_prompts.splitlines() if line.strip()]
        if not cleaned_model_path:
            st.error("Select an ONNX model before running batched generation.")
        elif not prompts:
            st.error("Enter at least one prompt.")
        else:
            try:
                boundaries = (
                    tuple(
                        int(value.strip())
                        for value in batch_bucket_text.split(",")
                        if value.strip()
                    )
                    if batch_bucket_text.strip()
                    else None
                )
                batch_config = BatchConfig(
                    max_batch_size=int(batch_max_size),
                    max_total_batch_tokens=int(batch_total_tokens),
                    batching_window_ms=float(batch_window_ms),
                    max_padding_ratio=float(batch_padding_ratio),
                    prompt_length_buckets=boundaries,
                )
            except (TypeError, ValueError) as exc:
                st.error(f"Invalid batching configuration: {exc}")
            else:
                with st.spinner("Loading model and executing dynamic batches..."):
                    try:
                        generator = OnnxTextGenerator(
                            Path(cleaned_model_path).parent,
                            model_path=cleaned_model_path,
                        )
                        requests = [
                            GenerationRequest(
                                prompt=prompt,
                                config=GenerationConfig(
                                    max_new_tokens=int(batch_max_new_tokens),
                                    use_cache=batch_cache,
                                ),
                                request_id=f"request-{index + 1}",
                            )
                            for index, prompt in enumerate(prompts)
                        ]
                        batcher = DynamicBatcher(
                            lambda pending: generator.generate_batch(
                                list(pending), batch_config
                            ),
                            token_counter=lambda request: generator.count_prompt_tokens(
                                request.prompt
                            ),
                            config=batch_config,
                        )

                        async def submit_all():
                            try:
                                return await asyncio.gather(
                                    *(batcher.submit(request) for request in requests)
                                )
                            finally:
                                await batcher.close()

                        results = asyncio.run(submit_all())
                        metrics = batcher.last_batch_metrics
                    except Exception as exc:
                        st.error(
                            f"Batched generation failed: {type(exc).__name__}: {exc}"
                        )
                    else:
                        st.success(
                            f"Completed {len(results)} requests in "
                            f"{len(metrics)} formed batch(es)."
                        )
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric(
                                "Effective Batch Size",
                                f"{sum(m.effective_batch_size for m in metrics) / len(metrics):.1f}"
                                if metrics else "N/A",
                            )
                        with metric_col2:
                            st.metric(
                                "Padding Overhead",
                                f"{sum(m.padding_overhead_ratio for m in metrics) / len(metrics) * 100:.1f}%"
                                if metrics else "N/A",
                            )
                        with metric_col3:
                            st.metric(
                                "Generated Tokens",
                                sum(result.generated_token_count for result in results),
                            )
                        with metric_col4:
                            st.metric(
                                "Aggregate Throughput",
                                fmt_float(
                                    sum(m.aggregate_tokens_per_second for m in metrics)
                                    if metrics else None,
                                    "tokens/s",
                                ),
                            )

                        st.markdown("### Results")
                        st.dataframe(
                            [
                                {
                                    "Request": result.request_id or index + 1,
                                    "Prompt Tokens": result.prompt_token_count,
                                    "Generated Tokens": result.generated_token_count,
                                    "TTFT": fmt_ms(result.time_to_first_token_ms),
                                    "Output": result.text,
                                }
                                for index, result in enumerate(results)
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.markdown("### Formed Batch Metrics")
                        st.dataframe(
                            [
                                {
                                    "Batch Size": metric.effective_batch_size,
                                    "Prompt Tokens": metric.prompt_tokens,
                                    "Padded Tokens": metric.padded_prompt_tokens,
                                    "Padding Overhead": f"{metric.padding_overhead_ratio * 100:.1f}%",
                                    "Formation": fmt_ms(metric.batch_formation_time_ms),
                                    "Execution": fmt_ms(metric.execution_time_ms),
                                    "Throughput": fmt_float(
                                        metric.aggregate_tokens_per_second,
                                        "tokens/s",
                                    ),
                                }
                                for metric in metrics
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )


# Mode 2: AutoTune Optimization
# =========================================================================
elif workflow_mode == "AutoTune Optimization":
    st.subheader("2. AutoTune Configuration")
    st.info(
        "AutoTune systematically evaluates candidate graph optimization levels "
        "(`disable`, `basic`, `extended`, `all`) and CPU thread counts against an "
        "unoptimized baseline (`graph_optimization='disable'`, `threads=1`). "
        "Only candidates with numerically correct outputs are eligible for selection."
    )

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        target_metric = st.selectbox(
            "Optimization Target Metric",
            options=["latency_p50_ms", "throughput_per_sec"],
            format_func=lambda m: "Median Latency (P50 - Lower is better)" if m == "latency_p50_ms" else "Throughput (Higher is better)",
            help="Metric to optimize when selecting the best candidate configuration",
        )
        execution_provider = st.selectbox(
            "Execution Provider",
            options=["CPUExecutionProvider"],
            index=0,
            help="ONNX Runtime execution provider for AutoTune",
        )
    with t_col2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            value=1,
            step=1,
            help="Inference batch size for candidate evaluations",
        )
    with t_col3:
        warmup_runs = st.number_input(
            "Warmup Runs per Candidate",
            min_value=0,
            value=5,
            step=1,
            help="Warmup iterations before timing each candidate",
        )
        timed_runs = st.number_input(
            "Timed Runs per Candidate",
            min_value=1,
            value=20,
            step=1,
            help="Timed iterations per candidate",
        )

    use_quantization = st.checkbox(
        "Include INT8 quantization candidates",
        value=False,
        help=(
            "Builds INT8 versions of the model and keeps one only if its outputs stay close "
            "to FP32 on your calibration samples."
        ),
    )
    calibration_path = ""
    min_cosine = 0.99
    min_top1 = 0.90
    if use_quantization:
        q_col1, q_col2, q_col3 = st.columns(3)
        with q_col1:
            calibration_path = st.text_input(
                "Calibration samples (.npz)",
                value="calibration/samples.npz",
                help="Real input samples, path relative to where the API server runs",
            )
        with q_col2:
            min_cosine = st.number_input(
                "Min cosine similarity", min_value=0.0, max_value=1.0,
                value=0.99, step=0.001, format="%.3f",
            )
        with q_col3:
            min_top1 = st.number_input(
                "Min top-1 agreement", min_value=0.0, max_value=1.0,
                value=0.90, step=0.01, format="%.2f",
            )

    with st.expander("Advanced Settings (Tolerances & Random Seed)"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            rel_tol = st.number_input(
                "Relative Tolerance (rel_tol)",
                min_value=0.0,
                value=1e-3,
                format="%.1e",
                help="Relative tolerance for output verification against baseline",
            )
        with adv_col2:
            abs_tol = st.number_input(
                "Absolute Tolerance (abs_tol)",
                min_value=0.0,
                value=1e-5,
                format="%.1e",
                help="Absolute tolerance for output verification against baseline",
            )
        with adv_col3:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=0,
                step=1,
                help="Random seed for input generation",
            )

    st.subheader("3. Execution & Results")
    tune_btn = st.button("Run AutoTune Optimization", type="primary", use_container_width=True)

    if tune_btn:
        cleaned_path = model_path.strip() if model_path else ""
        if not cleaned_path:
            st.error("Please provide a model path before running AutoTune.")
        elif not health.ok:
            st.error(f"Cannot run AutoTune: FastAPI backend at `{backend_url}` is offline.")
        else:
            payload = {
                "model_path": cleaned_path,
                "execution_provider": execution_provider,
                "metric": target_metric,
                "batch_size": int(batch_size),
                "warmup_runs": int(warmup_runs),
                "timed_runs": int(timed_runs),
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "seed": int(seed),
            }
            if use_quantization:
                payload.update({
                    "include_quantization": True,
                    "calibration_data_path": calibration_path.strip(),
                    "min_cosine": float(min_cosine),
                    "min_top1": float(min_top1),
                })

            with st.spinner("Running AutoTune candidate exploration and correctness verification..."):
                response = client.run_tune(payload)

            if not response.ok:
                st.error(
                    f"**AutoTune Request Failed"
                    + (f" (HTTP {response.status_code})" if response.status_code else "")
                    + f"**: {response.error}"
                )
            else:
                data = response.data or {}
                baseline = data.get("baseline") or {}
                best = data.get("best")
                speedup = data.get("speedup")
                eval_count = data.get("candidates_evaluated", 0)
                valid_count = data.get("candidates_valid", 0)
                candidates = data.get("candidates", [])
                backend_err = data.get("error_message")

                if backend_err:
                    st.error(f"**AutoTune Execution Failed**: {backend_err}")
                elif not best:
                    st.warning(
                        "**No Valid Candidates Found**: None of the evaluated candidate configurations "
                        "passed execution and correctness checks against the baseline."
                    )
                else:
                    metric_choice = data.get("metric", target_metric)
                    if metric_choice == "throughput_per_sec":
                        primary_label = "Throughput Change vs Baseline (%)"
                        tp_change = compute_throughput_change(baseline, best)
                        primary_val_str = format_throughput_change(tp_change)
                        if primary_val_str != "N/A":
                            success_msg = (
                                f"**AutoTune Complete!** Best candidate achieved **{primary_val_str}** "
                                f"Throughput Change vs Baseline ({valid_count}/{eval_count} candidates valid)."
                            )
                        else:
                            success_msg = (
                                f"**AutoTune Complete!** Best candidate selected by throughput "
                                f"({valid_count}/{eval_count} candidates valid)."
                            )
                    else:  # latency_p50_ms
                        primary_label = "Latency Speedup vs Baseline"
                        speedup_val = data.get("speedup")
                        if speedup_val is None:
                            speedup_val = compute_latency_speedup(baseline, best)
                        primary_val_str = format_latency_speedup(speedup_val)
                        if primary_val_str != "N/A":
                            success_msg = (
                                f"**AutoTune Complete!** Best candidate achieved **{primary_val_str}** "
                                f"Latency Speedup vs Baseline ({valid_count}/{eval_count} candidates valid)."
                            )
                        else:
                            success_msg = (
                                f"**AutoTune Complete!** Best candidate selected by median latency "
                                f"({valid_count}/{eval_count} candidates valid)."
                            )

                    st.success(success_msg)

                    # Top metrics
                    st.markdown("### Performance Comparison")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(primary_label, primary_val_str)
                    with c2:
                        st.metric("Baseline P50 Latency", fmt_ms(baseline.get("latency_p50_ms")))
                    with c3:
                        st.metric("Best P50 Latency", fmt_ms(best.get("latency_p50_ms")))
                    with c4:
                        st.metric("Best Throughput", fmt_float(best.get("throughput_per_sec"), "runs/s"))

                    # Best candidate summary
                    st.markdown("### Winning Configuration")
                    best_cfg = best.get("config", {})
                    w1, w2, w3, w4 = st.columns(4)
                    with w1:
                        st.write(f"**Graph Optimization:** `{best_cfg.get('graph_optimization_level')}`")
                    with w2:
                        st.write(f"**Intra-op Threads:** `{best_cfg.get('num_threads')}`")
                    with w3:
                        st.write(f"**Correctness:** `{best.get('correct')}` (Valid: `{best.get('is_valid')}`)")
                    with w4:
                        max_diff = best.get("max_abs_diff")
                        st.write(f"**Max Abs Diff:** `{max_diff:.2e}`" if isinstance(max_diff, (int, float)) else f"**Max Abs Diff:** `{max_diff}`")

                    best_meta = best_cfg.get("metadata") or {}
                    st.write(
                        f"**Precision:** `{best_meta.get('precision', 'fp32')}`"
                        + (f" (`{best_meta['method']}`)" if best_meta.get("method") else "")
                        + f"  |  **Model size:** {fmt_float(best.get('model_size_mb'), 'MB')}"
                    )
                    q_gain = data.get("quantization_speedup_vs_best_fp32")
                    if q_gain is not None:
                        st.caption(
                            f"Best INT8 vs best FP32 settings: {q_gain:.2f}x latency "
                            "(above 1.00x means INT8 is faster)."
                        )

                    # Candidates Table
                    st.markdown(f"### Candidate Exploration Summary ({valid_count}/{eval_count} Valid)")
                    rows = []
                    for idx, c in enumerate(candidates, start=1):
                        cfg = c.get("config", {})
                        meta = cfg.get("metadata") or {}
                        acc = c.get("accuracy") or {}
                        precision = meta.get("precision", "fp32")
                        if meta.get("method"):
                            precision = f"{precision} ({meta['method']})"
                        rows.append({
                            "#": str(idx),
                            "Precision": precision,
                            "Graph Opt": cfg.get("graph_optimization_level", "-"),
                            "Threads": cfg.get("num_threads", "-"),
                            "Valid": "Yes" if c.get("is_valid") else "❌ No",
                            "Correct": "Pass" if c.get("correct") else ("❌" if c.get("correct") is False else "-"),
                            "P50 Latency": fmt_ms(c.get("latency_p50_ms")),
                            "Mean Latency": fmt_ms(c.get("latency_mean_ms")),
                            "Throughput": fmt_float(c.get("throughput_per_sec"), "runs/s"),
                            "Peak Memory": fmt_float(c.get("peak_memory_mb"), "MB"),
                            "Size": fmt_float(c.get("model_size_mb"), "MB"),
                            "Top-1 Match": (
                                f"{acc['top1_agreement'] * 100:.0f}%"
                                if acc.get("top1_agreement") is not None else "-"
                            ),
                            "Cosine (min)": (
                                f"{acc['cosine_min']:.4f}"
                                if acc.get("cosine_min") is not None else "-"
                            ),
                        })
                    st.dataframe(rows, use_container_width=True)

                    # Environment details
                    env = best.get("environment", {})
                    with st.expander("Machine & Environment Information"):
                        if env:
                            e1, e2 = st.columns(2)
                            with e1:
                                st.write(f"**OS:** {env.get('os')} ({env.get('os_version')})")
                                st.write(f"**Machine / Arch:** {env.get('machine')} / {env.get('processor')}")
                                st.write(f"**Python Version:** {env.get('python')}")
                            with e2:
                                st.write(f"**ONNX Runtime Version:** {env.get('onnxruntime')}")
                                st.write(f"**Providers:** {', '.join(env.get('providers', []))}")

                    # JSON Download
                    stem = Path(cleaned_path).stem or "model"
                    st.download_button(
                        label="Download AutoTune Results (JSON)",
                        data=json.dumps(data, indent=2),
                        file_name=f"autotune_{stem}_{int(time.time())}.json",
                        mime="application/json",
                        use_container_width=True,
                    )


# =========================================================================
# Mode 2: Single Benchmark
# =========================================================================
else:
    st.subheader("2. Benchmark Configuration")
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        execution_provider = st.selectbox(
            "Execution Provider",
            options=["CPUExecutionProvider", "CUDAExecutionProvider", "CoreMLExecutionProvider"],
            index=0,
            help="ONNX Runtime execution provider",
        )
        graph_optimization_level = st.selectbox(
            "Graph Optimization Level",
            options=["all", "basic", "extended", "disable"],
            index=0,
            help="Level of ONNX graph optimization to apply",
        )

    with col_cfg2:
        num_threads = st.number_input(
            "Intra-op Threads",
            min_value=0,
            value=0,
            step=1,
            help="Thread count for intra-op parallelism (0 = let ONNX Runtime choose)",
        )
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            value=1,
            step=1,
            help="Inference batch size",
        )

    with col_cfg3:
        warmup_runs = st.number_input(
            "Warmup Runs",
            min_value=0,
            value=10,
            step=1,
            help="Untimed warmup iterations before measurement",
        )
        timed_runs = st.number_input(
            "Timed Runs",
            min_value=1,
            value=100,
            step=1,
            help="Number of iterations timed for latency statistics",
        )

    with st.expander("Advanced Settings (Tolerances & Random Seed)"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            rel_tol = st.number_input(
                "Relative Tolerance (rel_tol)",
                min_value=0.0,
                value=1e-3,
                format="%.1e",
                help="Relative tolerance for output array comparison against baseline",
            )
        with adv_col2:
            abs_tol = st.number_input(
                "Absolute Tolerance (abs_tol)",
                min_value=0.0,
                value=1e-5,
                format="%.1e",
                help="Absolute tolerance for output array comparison against baseline",
            )
        with adv_col3:
            seed = st.number_input(
                "Seed",
                min_value=0,
                value=0,
                step=1,
                help="Random seed for dummy input tensor generation",
            )

    st.subheader("3. Execution & Results")
    run_btn = st.button("Run Benchmark", type="primary", use_container_width=True)

    if run_btn:
        cleaned_path = model_path.strip() if model_path else ""
        if not cleaned_path:
            st.error("Please provide a model path before running the benchmark.")
        elif not health.ok:
            st.error(f"Cannot run benchmark: FastAPI backend at `{backend_url}` is offline.")
        else:
            payload = {
                "model_path": cleaned_path,
                "execution_provider": execution_provider,
                "graph_optimization_level": graph_optimization_level,
                "num_threads": int(num_threads),
                "batch_size": int(batch_size),
                "warmup_runs": int(warmup_runs),
                "timed_runs": int(timed_runs),
                "rel_tol": float(rel_tol),
                "abs_tol": float(abs_tol),
                "seed": int(seed),
            }

            with st.spinner("Executing benchmark on backend..."):
                response = client.run_benchmark(payload)

            if not response.ok:
                st.error(
                    f"**API Request Failed"
                    + (f" (HTTP {response.status_code})" if response.status_code else "")
                    + f"**: {response.error}"
                )
            else:
                result = response.data or {}
                success = result.get("success", False)
                is_valid = result.get("is_valid", False)
                correct = result.get("correct")
                error_message = result.get("error_message")

                # Status Banner - clearly distinguishing success vs is_valid
                if not success:
                    st.error(f"**Benchmark Failed During Execution**: {error_message or 'Unknown backend error'}")
                elif not is_valid:
                    st.warning(
                        "**Benchmark Completed with Invalid Output**: The execution completed (`success=True`), "
                        f"but correctness verification failed (`correct={correct}`). The candidate's output diverged "
                        "from the baseline run, so `is_valid=False`."
                    )
                else:
                    st.success(
                        "**Benchmark Successful and Valid**: The run finished without errors (`success=True`) "
                        f"and output values matched baseline within tolerances (`correct={correct}`, `is_valid=True`)."
                    )

                # Metric Cards
                st.markdown("### Performance Metrics")
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)

                with m_col1:
                    st.metric("Latency (Mean)", fmt_ms(result.get("latency_mean_ms")))
                    st.metric("Throughput", fmt_float(result.get("throughput_per_sec"), "runs/s"))

                with m_col2:
                    st.metric("Latency (Median / P50)", fmt_ms(result.get("latency_p50_ms")))
                    st.metric("Peak Memory", fmt_float(result.get("peak_memory_mb"), "MB"))

                with m_col3:
                    st.metric("Latency (P95)", fmt_ms(result.get("latency_p95_ms")))
                    st.metric("Correctness Gate", f"{correct}" if correct is not None else "N/A")

                with m_col4:
                    st.metric("Latency (P99)", fmt_ms(result.get("latency_p99_ms")))
                    max_diff = result.get("max_abs_diff")
                    diff_str = f"{max_diff:.2e}" if isinstance(max_diff, (int, float)) else "N/A"
                    st.metric("Max Abs Diff", diff_str)

                # Detailed Status & Range
                st.markdown("### Status Summary")
                st_col1, st_col2, st_col3, st_col4 = st.columns(4)
                with st_col1:
                    st.write(f"**Execution Success:** `{success}`")
                with st_col2:
                    st.write(f"**Result Is Valid:** `{is_valid}`")
                with st_col3:
                    st.write(f"**Min Latency:** {fmt_ms(result.get('latency_min_ms'))}")
                with st_col4:
                    st.write(f"**Max Latency:** {fmt_ms(result.get('latency_max_ms'))}")

                if error_message:
                    st.error(f"**Error Message:** {error_message}")

                # Expandable Environment Details
                env = result.get("environment", {})
                with st.expander("Machine & Environment Information"):
                    if env:
                        e_col1, e_col2 = st.columns(2)
                        with e_col1:
                            st.write(f"**OS:** {env.get('os')} ({env.get('os_version')})")
                            st.write(f"**Machine / Arch:** {env.get('machine')} / {env.get('processor')}")
                            st.write(f"**Python Version:** {env.get('python')}")
                        with e_col2:
                            st.write(f"**ONNX Runtime Version:** {env.get('onnxruntime')}")
                            st.write(f"**Providers:** {', '.join(env.get('providers', []))}")
                    else:
                        st.write("No environment data returned by backend.")

                # JSON Download Button
                stem = Path(cleaned_path).stem or "model"
                st.download_button(
                    label="Download Benchmark Result (JSON)",
                    data=json.dumps(result, indent=2),
                    file_name=f"benchmark_{stem}_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True,
                )
"""Streamlit dashboard for InferBench AutoTune."""

import json
from pathlib import Path
import time
import streamlit as st

from inferbench.dashboard.api_client import (
    DEFAULT_API_URL,
    InferBenchApiClient,
    discover_models,
)

# Page configuration
st.set_page_config(
    page_title="InferBench AutoTune",
    page_icon="⚡",
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
    st.sidebar.success(f"🟢 **Connected**\n\nService: `{health.service}`\nStatus: `{health.status}`")
else:
    st.sidebar.error(f"🔴 **Offline / Unreachable**\n\n{health.error}")
    st.warning(
        f"⚠️ **FastAPI backend is unreachable at `{backend_url}`**.\n\n"
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
        "ℹ️ No `.onnx` models found in the `models/` directory. "
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

# Configuration Controls
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

# Benchmark Execution
st.subheader("3. Execution & Results")
run_btn = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)

if run_btn:
    cleaned_path = model_path.strip() if model_path else ""
    if not cleaned_path:
        st.error("❌ Please provide a model path before running the benchmark.")
    elif not health.ok:
        st.error(f"❌ Cannot run benchmark: FastAPI backend at `{backend_url}` is offline.")
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
                f"❌ **API Request Failed"
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
                st.error(f"❌ **Benchmark Failed During Execution**: {error_message or 'Unknown backend error'}")
            elif not is_valid:
                st.warning(
                    "⚠️ **Benchmark Completed with Invalid Output**: The execution completed (`success=True`), "
                    f"but correctness verification failed (`correct={correct}`). The candidate's output diverged "
                    "from the baseline run, so `is_valid=False`."
                )
            else:
                st.success(
                    "✅ **Benchmark Successful and Valid**: The run finished without errors (`success=True`) "
                    f"and output values matched baseline within tolerances (`correct={correct}`, `is_valid=True`)."
                )

            # Metric Cards
            st.markdown("### Performance Metrics")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            def fmt_ms(val):
                return f"{val:.2f} ms" if isinstance(val, (int, float)) else "N/A"

            def fmt_float(val, unit=""):
                return f"{val:.1f} {unit}".strip() if isinstance(val, (int, float)) else "N/A"

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
            with st.expander("🖥️ Machine & Environment Information"):
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
                label="📥 Download Benchmark Result (JSON)",
                data=json.dumps(result, indent=2),
                file_name=f"benchmark_{stem}_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True,
            )

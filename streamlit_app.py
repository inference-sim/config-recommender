#!/usr/bin/env python3
"""Streamlit UI for GPU Recommendation Engine.

This application provides an interactive interface for GPU recommendation,
allowing users to input model specs and GPU catalog data, invoke the
recommendation API, and visualize results.
"""

import json
from io import StringIO
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from config_recommender import GPURecommender, GPUSpec, ModelArchitecture
from config_recommender.estimator import SyntheticBenchmarkEstimator

# Page configuration
st.set_page_config(
    page_title="GPU Recommendation Engine",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main Streamlit application."""
    st.markdown('<h1 class="main-header">🖥️ GPU Recommendation Engine</h1>', unsafe_allow_html=True)
    st.markdown(
        "**Find the optimal GPU for your ML model based on synthetic performance estimates**"
    )

    # Initialize session state
    if "models" not in st.session_state:
        st.session_state.models = []
    if "gpus" not in st.session_state:
        st.session_state.gpus = []
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Estimator parameters
        st.subheader("Performance Parameters")
        precision = st.selectbox(
            "Precision",
            options=["FP16", "FP32"],
            index=0,
            help="Model precision (FP16 = 2 bytes, FP32 = 4 bytes per parameter)",
        )
        precision_bytes = 2 if precision == "FP16" else 4

        batch_size = st.number_input(
            "Concurrent Users",
            min_value=1,
            max_value=100,
            value=1,
            help="Number of concurrent users (affects KV cache memory)",
        )

        compute_efficiency = st.slider(
            "Compute Efficiency",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Fraction of peak compute actually achieved (default: 50%)",
        )

        memory_overhead = st.slider(
            "Memory Overhead Factor",
            min_value=1.0,
            max_value=2.0,
            value=1.2,
            step=0.05,
            help="Memory overhead multiplier (default: 1.2 = 20% overhead)",
        )

        latency_bound = st.number_input(
            "Max Latency (ms/token)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Maximum acceptable latency per token (0 = no limit)",
        )
        latency_bound_ms = latency_bound if latency_bound > 0 else None

        st.divider()

        # Help & Info
        with st.expander("ℹ️ About"):
            st.markdown(
                """
                **GPU Recommendation Engine**
                
                This tool helps you find the optimal GPU for running ML models
                using synthetic benchmark estimation.
                
                **Features:**
                - Automatic model info fetching from HuggingFace
                - Memory analysis (weights, KV cache)
                - Performance prediction (throughput, latency)
                - Tensor parallelism support
                - Export results to JSON/CSV
                
                **How it works:**
                1. Add models and GPUs
                2. Configure performance parameters
                3. Click "Get Recommendations"
                4. View and export results
                """
            )

        with st.expander("🔍 Assumptions"):
            st.markdown(
                f"""
                **Current Configuration:**
                - Precision: {precision} ({precision_bytes} bytes/param)
                - Concurrent Users: {batch_size}
                - Compute Efficiency: {compute_efficiency * 100:.0f}%
                - Memory Overhead: {memory_overhead}x
                - Latency Bound: {latency_bound_ms if latency_bound_ms else 'None'}
                
                **Estimation Method:**
                - FLOPs-based compute throughput
                - Memory bandwidth-based throughput
                - Bottleneck analysis (compute vs memory)
                - Tensor parallelism for large models
                """
            )

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📊 Recommendations", "🤖 Models", "🖥️ GPUs"])

    with tab2:
        render_models_tab()

    with tab3:
        render_gpus_tab()

    with tab1:
        render_recommendations_tab(
            precision_bytes, batch_size, compute_efficiency, memory_overhead, latency_bound_ms
        )


def render_models_tab():
    """Render the models input tab."""
    st.markdown('<h2 class="section-header">Model Configuration</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Add Model")

        input_method = st.radio(
            "Input Method", ["Manual Entry", "JSON Upload"], horizontal=True, key="model_input_method"
        )

        if input_method == "Manual Entry":
            with st.form("add_model_form"):
                model_name = st.text_input(
                    "HuggingFace Model ID",
                    placeholder="e.g., Qwen/Qwen2.5-7B",
                    help="Enter the HuggingFace model identifier",
                )

                # Optional overrides for gated models
                with st.expander("Advanced: Manual Override (for gated models)"):
                    st.info(
                        "If you don't have a HuggingFace token for gated models, "
                        "you can manually specify parameters."
                    )
                    num_params = st.number_input(
                        "Parameters (Billions)", min_value=0.0, value=0.0, step=0.1
                    )
                    num_layers = st.number_input("Number of Layers", min_value=0, value=0, step=1)
                    hidden_size = st.number_input("Hidden Size", min_value=0, value=0, step=64)
                    num_heads = st.number_input("Attention Heads", min_value=0, value=0, step=1)
                    vocab_size = st.number_input("Vocab Size", min_value=0, value=0, step=1000)

                submitted = st.form_submit_button("Add Model", type="primary")

                if submitted and model_name:
                    try:
                        # Create model with optional overrides
                        model_kwargs = {"name": model_name}
                        if num_params > 0:
                            model_kwargs.update(
                                {
                                    "num_parameters": num_params,
                                    "num_layers": num_layers,
                                    "hidden_size": hidden_size,
                                    "num_attention_heads": num_heads,
                                    "vocab_size": vocab_size,
                                }
                            )

                        model = ModelArchitecture(**model_kwargs)
                        st.session_state.models.append(model)
                        st.success(f"✅ Added model: {model_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error adding model: {str(e)}")

        else:  # JSON Upload
            uploaded_file = st.file_uploader(
                "Upload Models JSON",
                type=["json"],
                help="Upload a JSON file with model specifications",
                key="models_json_upload",
            )

            if uploaded_file is not None:
                try:
                    models_data = json.load(uploaded_file)
                    if st.button("Load Models from JSON", key="load_models_json"):
                        loaded_count = 0
                        for model_dict in models_data:
                            try:
                                model = ModelArchitecture(**model_dict)
                                st.session_state.models.append(model)
                                loaded_count += 1
                            except Exception as e:
                                st.warning(f"Skipped model {model_dict.get('name', 'unknown')}: {e}")
                        st.success(f"✅ Loaded {loaded_count} models")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error parsing JSON: {str(e)}")

            st.markdown("**Example JSON format:**")
            st.code(
                """[
  {"name": "Qwen/Qwen2.5-7B"},
  {"name": "mistralai/Mixtral-8x7B-v0.1"}
]""",
                language="json",
            )

    with col2:
        st.subheader("Current Models")
        if st.session_state.models:
            for idx, model in enumerate(st.session_state.models):
                with st.container():
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.text(f"📄 {model.name}")
                        st.caption(f"{model.get_num_parameters():.1f}B parameters")
                    with cols[1]:
                        if st.button("🗑️", key=f"delete_model_{idx}"):
                            st.session_state.models.pop(idx)
                            st.rerun()
            if st.button("Clear All Models", key="clear_models"):
                st.session_state.models = []
                st.rerun()
        else:
            st.info("No models added yet")


def render_gpus_tab():
    """Render the GPUs input tab."""
    st.markdown('<h2 class="section-header">GPU Configuration</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Add GPU")

        input_method = st.radio(
            "Input Method", ["Manual Entry", "JSON Upload"], horizontal=True, key="gpu_input_method"
        )

        if input_method == "Manual Entry":
            with st.form("add_gpu_form"):
                gpu_name = st.text_input(
                    "GPU Name", placeholder="e.g., NVIDIA A100 80GB", help="Name/model of the GPU"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    memory_gb = st.number_input("Memory (GB)", min_value=1.0, value=80.0, step=1.0)
                    tflops_fp16 = st.number_input("TFLOPS FP16", min_value=1.0, value=312.0, step=1.0)
                    cost_per_hour = st.number_input(
                        "Cost ($/hour)", min_value=0.0, value=3.67, step=0.01
                    )

                with col_b:
                    bandwidth = st.number_input(
                        "Memory Bandwidth (GB/s)", min_value=1.0, value=2039.0, step=1.0
                    )
                    tflops_fp32 = st.number_input("TFLOPS FP32", min_value=1.0, value=156.0, step=1.0)

                submitted = st.form_submit_button("Add GPU", type="primary")

                if submitted and gpu_name:
                    gpu = GPUSpec(
                        name=gpu_name,
                        memory_gb=memory_gb,
                        memory_bandwidth_gb_s=bandwidth,
                        tflops_fp16=tflops_fp16,
                        tflops_fp32=tflops_fp32,
                        cost_per_hour=cost_per_hour,
                    )
                    st.session_state.gpus.append(gpu)
                    st.success(f"✅ Added GPU: {gpu_name}")
                    st.rerun()

        else:  # JSON Upload
            uploaded_file = st.file_uploader(
                "Upload GPUs JSON",
                type=["json"],
                help="Upload a JSON file with GPU specifications",
                key="gpus_json_upload",
            )

            if uploaded_file is not None:
                try:
                    gpus_data = json.load(uploaded_file)
                    if st.button("Load GPUs from JSON", key="load_gpus_json"):
                        loaded_count = 0
                        for gpu_dict in gpus_data:
                            try:
                                gpu = GPUSpec(**gpu_dict)
                                st.session_state.gpus.append(gpu)
                                loaded_count += 1
                            except Exception as e:
                                st.warning(f"Skipped GPU {gpu_dict.get('name', 'unknown')}: {e}")
                        st.success(f"✅ Loaded {loaded_count} GPUs")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error parsing JSON: {str(e)}")

            st.markdown("**Example JSON format:**")
            st.code(
                """[
  {
    "name": "NVIDIA A100 80GB",
    "memory_gb": 80.0,
    "memory_bandwidth_gb_s": 2039.0,
    "tflops_fp16": 312.0,
    "tflops_fp32": 156.0,
    "cost_per_hour": 3.67
  }
]""",
                language="json",
            )

    with col2:
        st.subheader("Current GPUs")
        if st.session_state.gpus:
            for idx, gpu in enumerate(st.session_state.gpus):
                with st.container():
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.text(f"🖥️ {gpu.name}")
                        st.caption(f"{gpu.memory_gb}GB • {gpu.tflops_fp16} TFLOPS FP16")
                    with cols[1]:
                        if st.button("🗑️", key=f"delete_gpu_{idx}"):
                            st.session_state.gpus.pop(idx)
                            st.rerun()
            if st.button("Clear All GPUs", key="clear_gpus"):
                st.session_state.gpus = []
                st.rerun()
        else:
            st.info("No GPUs added yet")


def render_recommendations_tab(
    precision_bytes: int,
    batch_size: int,
    compute_efficiency: float,
    memory_overhead: float,
    latency_bound_ms: Optional[float],
):
    """Render the recommendations results tab."""
    st.markdown('<h2 class="section-header">GPU Recommendations</h2>', unsafe_allow_html=True)

    # Check if we have models and GPUs
    if not st.session_state.models:
        st.warning("⚠️ Please add models in the 'Models' tab first")
        return

    if not st.session_state.gpus:
        st.warning("⚠️ Please add GPUs in the 'GPUs' tab first")
        return

    # Button to generate recommendations
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating recommendations..."):
            try:
                estimator = SyntheticBenchmarkEstimator(
                    precision_bytes=precision_bytes,
                    memory_overhead_factor=memory_overhead,
                    compute_efficiency=compute_efficiency,
                    concurrent_users=batch_size,
                )
                recommender = GPURecommender(estimator=estimator, latency_bound_ms=latency_bound_ms)

                results = recommender.recommend_for_models(
                    st.session_state.models, st.session_state.gpus
                )
                st.session_state.recommendations = results
                st.success("✅ Recommendations generated!")
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                return

    # Display recommendations if available
    if st.session_state.recommendations:
        display_recommendations(st.session_state.recommendations)


def display_recommendations(recommendations: List):
    """Display recommendation results in a table and detail view."""
    st.divider()

    # Summary metrics
    st.subheader("📈 Summary")
    col1, col2, col3, col4 = st.columns(4)

    total_models = len(recommendations)
    models_with_gpu = sum(1 for r in recommendations if r.recommended_gpu is not None)

    with col1:
        st.metric("Total Models", total_models)
    with col2:
        st.metric("Models with GPU", models_with_gpu)
    with col3:
        avg_throughput = (
            sum(
                r.performance.tokens_per_second
                for r in recommendations
                if r.performance and r.recommended_gpu
            )
            / models_with_gpu
            if models_with_gpu > 0
            else 0
        )
        st.metric("Avg Throughput", f"{avg_throughput:.0f} tok/s")
    with col4:
        avg_latency = (
            sum(
                r.performance.intertoken_latency_ms
                for r in recommendations
                if r.performance and r.recommended_gpu
            )
            / models_with_gpu
            if models_with_gpu > 0
            else 0
        )
        st.metric("Avg Latency", f"{avg_latency:.2f} ms")

    st.divider()

    # Create DataFrame for table view
    table_data = []
    for rec in recommendations:
        row = {
            "Model": rec.model_name,
            "Recommended GPU": rec.recommended_gpu or "None",
            "Throughput (tok/s)": (
                f"{rec.performance.tokens_per_second:.1f}" if rec.performance else "N/A"
            ),
            "Latency (ms)": (
                f"{rec.performance.intertoken_latency_ms:.2f}" if rec.performance else "N/A"
            ),
            "Memory (GB)": f"{rec.performance.memory_required_gb:.1f}" if rec.performance else "N/A",
            "Fits": "✅" if rec.performance and rec.performance.fits_in_memory else "❌",
            "Bottleneck": (
                "Compute" if rec.performance and rec.performance.compute_bound else "Memory"
            )
            if rec.performance
            else "N/A",
            "TP Size": (
                rec.performance.tensor_parallel_size if rec.performance else 1
            ),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Filters
    st.subheader("🔍 Filter & Sort")
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_gpu = st.multiselect(
            "Filter by GPU",
            options=df["Recommended GPU"].unique(),
            default=df["Recommended GPU"].unique(),
        )
    with col2:
        filter_bottleneck = st.multiselect(
            "Filter by Bottleneck",
            options=df["Bottleneck"].unique(),
            default=df["Bottleneck"].unique(),
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["Model", "Throughput (tok/s)", "Latency (ms)", "Memory (GB)"],
            index=1,
        )

    # Apply filters
    filtered_df = df[
        (df["Recommended GPU"].isin(filter_gpu)) & (df["Bottleneck"].isin(filter_bottleneck))
    ]

    # Apply sorting
    if sort_by != "Model":
        # Convert to numeric for sorting (make a copy to avoid SettingWithCopyWarning)
        if sort_by in ["Throughput (tok/s)", "Latency (ms)", "Memory (GB)"]:
            filtered_df = filtered_df.copy()
            filtered_df[sort_by] = pd.to_numeric(
                filtered_df[sort_by].str.replace("N/A", "0"), errors="coerce"
            )
            filtered_df = filtered_df.sort_values(
                by=sort_by, ascending=(sort_by == "Latency (ms)")
            )

    # Display table
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    st.divider()

    # Detailed view for each model
    st.subheader("📋 Detailed Results")

    for idx, rec in enumerate(recommendations):
        with st.expander(f"**{rec.model_name}** → {rec.recommended_gpu or 'No compatible GPU'}"):
            if rec.recommended_gpu:
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Throughput", f"{rec.performance.tokens_per_second:.1f} tok/s"
                    )
                with col2:
                    st.metric(
                        "Latency", f"{rec.performance.intertoken_latency_ms:.2f} ms/token"
                    )
                with col3:
                    st.metric(
                        "Memory Used", f"{rec.performance.memory_required_gb:.1f} GB"
                    )
                with col4:
                    st.metric(
                        "TP Size", rec.performance.tensor_parallel_size
                    )

                # Memory breakdown
                st.markdown("**Memory Breakdown:**")
                mem_col1, mem_col2 = st.columns(2)
                with mem_col1:
                    st.write(f"- Weights: {rec.performance.memory_weights_gb:.2f} GB")
                with mem_col2:
                    st.write(f"- KV Cache: {rec.performance.memory_kv_cache_gb:.2f} GB")

                # Bottleneck analysis
                st.markdown("**Bottleneck Analysis:**")
                if rec.performance.compute_bound:
                    st.info("🔢 **Compute-bound**: Performance limited by GPU compute (FLOPs)")
                else:
                    st.info(
                        "💾 **Memory-bound**: Performance limited by memory bandwidth"
                    )

                # All compatible GPUs
                if rec.all_compatible_gpus:
                    st.markdown("**All Compatible GPUs:**")
                    compat_df = pd.DataFrame(rec.all_compatible_gpus)
                    if not compat_df.empty:
                        # Format numeric columns
                        if "tokens_per_second" in compat_df.columns:
                            compat_df["tokens_per_second"] = compat_df[
                                "tokens_per_second"
                            ].apply(lambda x: f"{x:.1f}")
                        if "intertoken_latency_ms" in compat_df.columns:
                            compat_df["intertoken_latency_ms"] = compat_df[
                                "intertoken_latency_ms"
                            ].apply(lambda x: f"{x:.2f}")
                        if "memory_required_gb" in compat_df.columns:
                            compat_df["memory_required_gb"] = compat_df[
                                "memory_required_gb"
                            ].apply(lambda x: f"{x:.1f}")
                        st.dataframe(compat_df, use_container_width=True, hide_index=True)

            # Reasoning
            st.markdown("**Reasoning:**")
            st.info(rec.reasoning)

    st.divider()

    # Export options
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        # Export to JSON
        json_data = json.dumps(
            {"recommendations": [rec.to_dict() for rec in recommendations]}, indent=2
        )
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name="gpu_recommendations.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        # Export to CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name="gpu_recommendations.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

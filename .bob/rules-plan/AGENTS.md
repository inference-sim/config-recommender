# Plan Mode Rules (Non-Obvious Only)

## Architectural Constraints

### BentoML Integration is Intentionally Hybrid
- Performance estimation uses BentoML's llm-optimizer (roofline analysis)
- Memory calculation uses config_explorer (from llm-d-benchmark)
- Fallback to custom calculation is INTENTIONAL for unsupported GPUs (V100, T4)
- This hybrid approach provides best accuracy across all GPU types

### Tensor Parallelism Discovery Pattern
- TP is only attempted when single GPU fails (see [`recommender.py:110`](config_recommender/recommender.py:110))
- TP sizes come from `find_possible_tp()` in config_explorer, not hardcoded
- TP overhead (5% per rank) only applied in fallback calculation, not BentoML path
- This prevents unnecessary multi-GPU recommendations when single GPU suffices

### Model Information Fetching Strategy
- Production: Automatic HF fetching via config_explorer in [`ModelArchitecture.__post_init__`](config_recommender/models.py:67)
- Tests: Manual parameters to avoid network dependencies
- Gated models: Graceful fallback with helpful error messages
- This dual-mode design enables both convenience and testability

## Hidden Dependencies

### Private Field Lifecycle
- `_model_info`, `_model_config`, `_kv_cache_detail` in ModelArchitecture are lazy-loaded
- Populated in `__post_init__`, never set directly
- `_kv_cache_detail` is cached and reused for same context_len/batch_size
- This pattern optimizes repeated calculations while maintaining encapsulation

### GPU Library vs Custom GPU Separation
- GPU library uses simplified keys ("H100") for user convenience
- GPUSpec uses full names ("NVIDIA H100 80GB") for clarity
- CLI enforces mutual exclusivity between `--gpus` and `--gpu-library`
- `--extend-gpus` only works with `--gpu-library` to maintain clear semantics

## Performance Bottleneck Patterns

### KV Cache Batch Size Convention
- Single-user scenario uses `batch_size=1` at max_model_len
- This differs from typical batch inference patterns
- Represents realistic single concurrent request scenario
- See [`estimator.py:168`](config_recommender/estimator.py:168)

### Sequence Length Dual Purpose
- CLI: `kv_sequence_length = input_length + output_length` for KV cache sizing
- Estimator: Uses `input_length` and `output_length` separately for prefill/decode phases
- This distinction enables workload-specific estimation accuracy

## Extension Points

### Custom GPU Integration
- GPU library is a simple dict in [`gpu_library.py`](config_recommender/gpu_library.py)
- `create_custom_gpu()` function for programmatic GPU creation
- CLI supports both `--gpus` (file) and `--gpu-library` (preloaded) + `--extend-gpus`
- This flexibility supports both convenience and customization

### Model Parameter Override
- Manual parameters bypass HF fetching for gated models
- Requires: `num_parameters`, `num_layers`, `hidden_size`, `num_attention_heads`, `vocab_size`
- Optional: `num_kv_heads` (defaults to `num_attention_heads`), `max_sequence_length`
- This enables usage without HF tokens while maintaining accuracy
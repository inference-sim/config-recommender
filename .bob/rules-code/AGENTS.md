# Code Mode Rules (Non-Obvious Only)

## Testing Patterns

### Manual Model Parameters Required
- Test models MUST provide all fields: `num_parameters`, `num_layers`, `hidden_size`, `num_attention_heads`, `vocab_size`, `max_sequence_length`
- Never rely on HuggingFace fetching in tests (network dependency)
- See [`tests/test_recommender.py:13`](tests/test_recommender.py:13) for fixture pattern

### KV Cache Batch Size Convention
- Always use `batch_size=1` for KV cache calculations in single-user scenarios
- This is NOT a bug - represents single concurrent request at max_model_len
- See [`estimator.py:168`](config_recommender/estimator.py:168)

## Model Architecture Patterns

### Private Field Initialization
- `_model_info`, `_model_config`, `_kv_cache_detail` in [`ModelArchitecture`](config_recommender/models.py:61) are `field(init=False, repr=False)`
- NEVER set these directly - they're populated in `__post_init__`
- Gated model errors check for keywords: 'gated', 'access', 'authentication', 'token', '401', '403' (line 80)

### Manual Override Validation
- When `num_parameters` is provided, MUST also provide: `num_layers`, `hidden_size`, `num_attention_heads`, `vocab_size`
- Missing fields raise ValueError with helpful message (line 92-96)
- `num_kv_heads` defaults to `num_attention_heads` if not provided (line 99-100)

## Estimator Patterns

### BentoML Fallback is Intentional
- ValueError at [`estimator.py:295`](config_recommender/estimator.py:295) is NOT an error - it's the fallback mechanism
- V100 and T4 deliberately excluded from `gpu_name_mapping` to use custom specs
- Fallback provides accurate results using our GPU specs

### Tensor Parallelism Overhead
- [`TP_OVERHEAD_PER_RANK = 0.05`](config_recommender/estimator.py:33) applied multiplicatively: `1.0 - (0.05 * (tp_size - 1))`
- Only applied in fallback calculation (line 333), not when using BentoML
- TP=2: 5%, TP=4: 15%, TP=8: 35% overhead

## CLI Patterns

### GPU Selection Mutually Exclusive
- `--gpus` and `--gpu-library` are mutually exclusive (line 76)
- `--extend-gpus` ONLY works with `--gpu-library`, not with `--gpus`
- GPU library uses simplified keys ("H100") vs full names ("NVIDIA H100 80GB")

### Sequence Length Calculation
- `kv_sequence_length = input_length + output_length` (line 182)
- This is for KV cache sizing, not performance estimation
- Performance uses `input_length` and `output_length` separately
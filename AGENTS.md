# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Non-Obvious Project Patterns

### Testing with Manual Model Parameters
- Tests use manual model parameters (num_parameters, num_layers, etc.) instead of HuggingFace fetching to avoid network dependencies
- When creating test models, MUST provide all required fields: num_parameters, num_layers, hidden_size, num_attention_heads, vocab_size, max_sequence_length
- Production code fetches from HF automatically, but tests bypass this

### BentoML Integration Fallback Pattern
- [`estimator.py:estimate_performance()`](config_recommender/estimator.py:196) has intentional fallback logic when GPU not in BentoML's supported list
- V100 and T4 deliberately excluded from `gpu_name_mapping` to trigger custom calculation using accurate specs
- ValueError catch at line 295 is NOT an error - it's the fallback mechanism

### Tensor Parallelism Overhead
- TP overhead constant [`TP_OVERHEAD_PER_RANK = 0.05`](config_recommender/estimator.py:33) is applied multiplicatively per rank beyond 1
- TP=2: 5% overhead, TP=4: 15%, TP=8: 35% - this is conservative estimate for activation communication
- Applied at line 333 only in fallback calculation (BentoML handles it internally)

### KV Cache Batch Size Convention
- KV cache calculations use `batch_size=1` for single request at max_model_len (see [`estimator.py:168`](config_recommender/estimator.py:168))
- This differs from typical batch inference - represents single concurrent user scenario
- CLI calculates `kv_sequence_length = input_length + output_length` at line 182

### Model Config Private Fields
- [`ModelArchitecture`](config_recommender/models.py:25) uses `field(init=False, repr=False)` for `_model_info`, `_model_config`, `_kv_cache_detail`
- These are populated in `__post_init__` and should never be set directly
- Gated model error handling at line 78 checks for specific keywords in error message

### GPU Library vs Custom GPU Distinction
- CLI has mutually exclusive group for `--gpus` vs `--gpu-library` (line 76)
- `--extend-gpus` only works WITH `--gpu-library`, not with `--gpus` alone
- GPU library keys are simplified (e.g., "H100", "A100-80GB") vs full names in GPUSpec

## Commands

```bash
# Run single test file
pytest tests/test_recommender.py

# Run tests with coverage (configured in pyproject.toml)
pytest --cov=config_recommender tests/

# Format code (line-length 100, target py311+)
black config_recommender/ tests/
isort config_recommender/ tests/

# CLI with sequence lengths (for workload-specific estimation)
config-recommender --models examples/models.json --gpu-library H100 \
    --input-length 1024 --output-length 512
```

## Code Style (from CLAUDE.md and .github/copilot-instructions.md)

- Black: line-length 100, target Python 3.11+
- isort: black profile, line-length 100
- Type hints required for functions and return values
- Docstrings for public functions, classes, modules
- Minimal comments - code should be self-documenting
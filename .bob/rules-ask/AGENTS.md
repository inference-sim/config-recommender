# Ask Mode Rules (Non-Obvious Only)

## Documentation Context

### Multiple Documentation Sources
- CLAUDE.md: Comprehensive project overview with architecture details
- README.md: User-facing documentation with examples
- .github/copilot-instructions.md: Code style guidelines
- AGENTS.md: Non-obvious patterns for AI assistants

### Counterintuitive Naming
- "Synthetic Benchmark" means FLOPs-based estimation, NOT actual benchmarking
- "Estimator" uses BentoML's llm-optimizer for roofline analysis, not simple calculations
- "GPU Library" is a preloaded dict, not an external library dependency

## Architecture Context

### BentoML Integration is Hybrid
- Uses BentoML's llm-optimizer for performance estimation (roofline analysis)
- Uses config_explorer (from llm-d-benchmark) for model memory calculations
- Fallback to custom calculation when GPU not in BentoML's list (intentional for V100/T4)

### Three Interfaces, One Engine
- Python API: Direct programmatic access via [`__init__.py`](config_recommender/__init__.py)
- CLI: [`cli.py`](config_recommender/cli.py) with argparse
- Streamlit UI: [`streamlit_app.py`](streamlit_app.py) for interactive use
- All use same core: [`recommender.py`](config_recommender/recommender.py) + [`estimator.py`](config_recommender/estimator.py)

## Hidden Behaviors

### Gated Model Handling
- [`ModelArchitecture.__post_init__`](config_recommender/models.py:67) tries HF fetch, catches errors
- Checks error message for keywords: 'gated', 'access', 'authentication', 'token', '401', '403'
- Provides helpful error message with two solutions: set HF_TOKEN or provide manual params

### Tensor Parallelism Auto-Discovery
- [`recommender.py:113`](config_recommender/recommender.py:113) uses `find_possible_tp()` from config_explorer
- Only tries TP when single GPU fails to fit model
- TP sizes come from capacity planner, not hardcoded list

### KV Cache Detail Object
- [`ModelArchitecture.get_kv_cache_detail()`](config_recommender/models.py:198) returns `KVCacheDetail` from config_explorer
- Only available when HF info successfully fetched (not for manual overrides)
- Cached and reused if context_len/batch_size unchanged

## File Organization

### Examples Directory Structure
- `examples/models.json`: Sample model configurations (HF identifiers only)
- `examples/custom_gpus.json`: Sample GPU specs (full specifications)
- `examples/*.py`: Usage patterns (basic, advanced, tensor parallelism, GPU library)
- `examples/README.md`: Detailed examples documentation

### Test Organization
- Tests in `tests/` directory, one file per module
- Test fixtures use manual parameters to avoid network calls
- No separate test data directory - fixtures defined in test files
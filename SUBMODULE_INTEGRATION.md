# Submodule Integration Guide

This document explains how to integrate `config-recommender` as a submodule in the `llm-d-benchmark` repository.

## Overview

The `config-recommender` repository is designed to work in two modes:
1. **Standalone mode**: Installed independently with all dependencies
2. **Submodule mode**: Integrated into `llm-d-benchmark` as a submodule

## Architecture

### Dependency Relationship

```
llm-d-benchmark/
├── config_explorer/        # Config exploration library
└── config-recommender/     # GPU recommendation engine (this repo)
    └── depends on config_explorer
```

When `config-recommender` is used as a submodule in `llm-d-benchmark`, it can access `config_explorer` from the parent repository's directory structure, avoiding the need to install it separately from Git.

## Integration Steps for llm-d-benchmark

### 1. Add config-recommender as a Submodule

From the root of the `llm-d-benchmark` repository:

```bash
# Add config-recommender as a submodule
git submodule add https://github.com/inference-sim/config-recommender.git config-recommender

# Initialize and update the submodule
git submodule update --init --recursive

# Commit the submodule addition
git add .gitmodules config-recommender
git commit -m "Add config-recommender as submodule"
```

### 2. Install Dependencies

After adding the submodule, install both `config_explorer` and `config-recommender`:

```bash
# From llm-d-benchmark root directory

# Install config_explorer (if not already installed)
pip install ./config_explorer

# Install config-recommender in submodule mode
# Option 1: Using setup.py (automatically detects submodule mode)
pip install ./config-recommender

# Option 2: Using requirements-submodule.txt (explicit submodule mode)
pip install -r config-recommender/requirements-submodule.txt
pip install -e ./config-recommender

# Option 3: Using environment variable to force submodule mode
CONFIG_RECOMMENDER_SUBMODULE_MODE=1 pip install ./config-recommender
```

The installation will automatically detect that it's in a submodule and skip installing `config_explorer` from Git, expecting it to be available from the parent repository.

### 3. Verify Installation

```python
# Test that both packages are available
import config_explorer
import config_recommender

# Test basic functionality
from config_recommender import ModelArchitecture, GPURecommender, get_gpu_specs

model = ModelArchitecture(name="Qwen/Qwen2.5-7B")
gpus = get_gpu_specs(["H100", "A100-80GB"])
recommender = GPURecommender()
result = recommender.recommend_gpu(model, gpus)

print(f"Recommended GPU: {result.recommended_gpu}")
```

### 4. Update the Repository

If you need to pull the latest changes from the submodule:

```bash
# Update to the latest commit
git submodule update --remote config-recommender

# Or enter the submodule directory and pull
cd config-recommender
git pull origin main
cd ..

# Commit the submodule update
git add config-recommender
git commit -m "Update config-recommender submodule"
```

## How It Works

### Automatic Submodule Detection

The `setup.py` file includes logic to automatically detect if it's being installed as a submodule:

```python
# Check if ../config_explorer exists (sibling directory in parent repo)
parent_dir = Path(__file__).parent.parent
config_explorer_path = parent_dir / "config_explorer"
is_submodule = config_explorer_path.exists() and (config_explorer_path / "pyproject.toml").exists()
```

When the above condition is true, `config-recommender` will:
- Skip installing `config_explorer` from Git
- Assume `config_explorer` is available from the parent repository
- Only install its other dependencies (`llm-optimizer`, `streamlit`, `pandas`)

### Manual Override

You can manually force submodule mode by setting an environment variable:

```bash
export CONFIG_RECOMMENDER_SUBMODULE_MODE=1
pip install ./config-recommender
```

## Usage in llm-d-benchmark

Once integrated as a submodule, you can use `config-recommender` in your `llm-d-benchmark` code:

```python
# In llm-d-benchmark scripts
from config_explorer.capacity_planner import get_model_info_from_hf
from config_recommender import ModelArchitecture, GPURecommender, get_gpu_specs

# Your benchmarking code can now use both libraries
model = ModelArchitecture(name="meta-llama/Llama-3.1-70B-Instruct")
gpus = get_gpu_specs(["H100", "H200"])

recommender = GPURecommender()
result = recommender.recommend_gpu(model, gpus)
```

## Standalone Usage

The repository continues to work perfectly in standalone mode:

```bash
# Clone the repository
git clone https://github.com/inference-sim/config-recommender.git
cd config-recommender

# Install in standalone mode (will fetch config_explorer from Git)
pip install -e .

# Use as normal
config-recommender --models examples/models.json --gpu-library H100 A100-80GB
```

## Testing

### Testing in Submodule Mode

From the `llm-d-benchmark` root:

```bash
# Install in development mode
pip install -e ./config_explorer
pip install -e ./config-recommender

# Run config-recommender tests
cd config-recommender
pytest tests/
cd ..
```

### Testing in Standalone Mode

From the `config-recommender` root:

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Troubleshooting

### config_explorer not found

If you get `ModuleNotFoundError: No module named 'config_explorer'`:

1. **In submodule mode**: Ensure `config_explorer` is installed from the parent repo
   ```bash
   pip install ./config_explorer
   ```

2. **In standalone mode**: The package should install automatically. If not:
   ```bash
   pip install "config_explorer @ git+https://github.com/llm-d/llm-d-benchmark.git#subdirectory=config_explorer"
   ```

### Submodule not detected

If the automatic detection isn't working:

1. Manually set the environment variable:
   ```bash
   export CONFIG_RECOMMENDER_SUBMODULE_MODE=1
   pip install ./config-recommender
   ```

2. Or use the explicit submodule requirements file:
   ```bash
   pip install -r config-recommender/requirements-submodule.txt
   pip install -e ./config-recommender
   ```

## Benefits of Submodule Integration

1. **Shared Development**: Changes to `config_explorer` are immediately available to `config-recommender`
2. **Version Synchronization**: Both packages stay in sync within the parent repository
3. **Simplified Testing**: Test both packages together in the same environment
4. **No Duplication**: Single source of truth for `config_explorer`
5. **Flexibility**: Repository continues to work standalone when needed

## Contributing

When making changes to `config-recommender` as a submodule:

1. Make changes in the submodule directory
2. Commit and push from within the submodule:
   ```bash
   cd config-recommender
   git add .
   git commit -m "Your changes"
   git push origin main
   cd ..
   ```
3. Update the parent repository to track the new commit:
   ```bash
   git add config-recommender
   git commit -m "Update config-recommender submodule"
   git push
   ```

## References

- [Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [llm-d-benchmark Repository](https://github.com/llm-d/llm-d-benchmark)
- [config_explorer Documentation](https://github.com/llm-d/llm-d-benchmark/tree/main/config_explorer)

# Integration Verification Checklist

This document provides a checklist for verifying the submodule integration works correctly.

## Pre-Integration Verification (Complete ✅)

- [x] Automatic submodule detection implemented
- [x] Dual-mode installation support (standalone + submodule)
- [x] Environment variable override option
- [x] Comprehensive documentation created
- [x] Test coverage implemented
- [x] Security scan completed (0 vulnerabilities)
- [x] Code review completed
- [x] All tests passing (6/6)

## Post-Integration Verification (For llm-d-benchmark maintainers)

After adding config-recommender as a submodule to llm-d-benchmark, verify:

### 1. Submodule Addition
```bash
# From llm-d-benchmark root
git submodule add https://github.com/inference-sim/config-recommender.git config-recommender
git submodule update --init --recursive
```

Verify:
- [ ] `.gitmodules` file created/updated
- [ ] `config-recommender` directory exists
- [ ] Submodule shows correct commit hash

### 2. Installation
```bash
# Install config_explorer first (if not already installed)
pip install ./config_explorer

# Install config-recommender
pip install ./config-recommender
```

Verify:
- [ ] Installation completes without errors
- [ ] config_explorer is NOT installed from git (should skip it)
- [ ] Other dependencies (llm-optimizer, streamlit, pandas) are installed

### 3. Import Testing
```python
# Test imports work correctly
import config_explorer
import config_recommender

from config_explorer.capacity_planner import get_model_info_from_hf
from config_recommender import ModelArchitecture, GPURecommender, get_gpu_specs

print("✓ All imports successful")
```

Verify:
- [ ] No import errors
- [ ] Both packages accessible

### 4. Functionality Testing
```python
# Test basic functionality
model = ModelArchitecture(name="Qwen/Qwen2.5-7B")
gpus = get_gpu_specs(["H100", "A100-80GB"])
recommender = GPURecommender()
result = recommender.recommend_gpu(model, gpus)

print(f"Recommended GPU: {result.recommended_gpu}")
print(f"Throughput: {result.performance.tokens_per_second:.2f} tokens/sec")
```

Verify:
- [ ] Code runs without errors
- [ ] Reasonable recommendation returned
- [ ] Both config_explorer and config-recommender working together

### 5. CLI Testing
```bash
# Test CLI works
cd config-recommender
config-recommender --list-gpus
config-recommender --models examples/models.json --gpu-library H100 A100-80GB
```

Verify:
- [ ] CLI executable found
- [ ] Commands run successfully
- [ ] Output looks correct

### 6. Update Testing
```bash
# Test submodule updates work
cd config-recommender
git pull origin main
cd ..
git add config-recommender
git commit -m "Update config-recommender submodule"
```

Verify:
- [ ] Submodule can be updated
- [ ] Changes can be committed
- [ ] No conflicts with parent repo

## Troubleshooting

If any verification fails, refer to:
- `SUBMODULE_INTEGRATION.md` - Comprehensive integration guide
- `README.md` - Installation instructions
- Test files for examples of correct usage

## Success Criteria

All checkboxes above should be checked (✓) for successful integration.

## Support

For issues or questions:
1. Check `SUBMODULE_INTEGRATION.md` troubleshooting section
2. Review test files for working examples
3. Open an issue in the config-recommender repository

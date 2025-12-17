"""Setup configuration for config_recommender package."""

import os
from pathlib import Path
from setuptools import setup, find_packages

# Git repository URLs (centralized for easy maintenance)
LLM_D_BENCHMARK_REPO = "https://github.com/llm-d/llm-d-benchmark.git"
LLM_OPTIMIZER_REPO = "https://github.com/bentoml/llm-optimizer.git"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Detect if we're being installed as a submodule in llm-d-benchmark
# This checks if ../config_explorer exists (sibling directory in parent repo)
parent_dir = Path(__file__).parent.parent
config_explorer_path = parent_dir / "config_explorer"
is_submodule = config_explorer_path.exists() and (config_explorer_path / "pyproject.toml").exists()

# Configure dependencies based on installation mode
if is_submodule or os.environ.get("CONFIG_RECOMMENDER_SUBMODULE_MODE") == "1":
    # Submodule mode: config_explorer should be installed from local path
    # The parent repository (llm-d-benchmark) should handle installing config_explorer
    install_requires = [
        f"llm-optimizer @ git+{LLM_OPTIMIZER_REPO}",
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
    ]
else:
    # Standalone mode: install config_explorer from git
    install_requires = [
        f"config_explorer @ git+{LLM_D_BENCHMARK_REPO}#subdirectory=config_explorer",
        f"llm-optimizer @ git+{LLM_OPTIMIZER_REPO}",
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
    ]

setup(
    name="config-recommender",
    version="0.1.0",
    author="Config Recommender Team",
    description="GPU recommendation engine for ML inference with synthetic benchmark estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "config-recommender=config_recommender.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

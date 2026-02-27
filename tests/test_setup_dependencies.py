#!/usr/bin/env python3
"""Test script to verify setup.py generates correct dependencies"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Constants for repository URLs (should match setup.py)
LLM_D_BENCHMARK_REPO = "https://github.com/llm-d/llm-d-benchmark.git"


def get_install_requires(setup_dir):
    """Extract install_requires from setup.py without actually running setup"""
    import importlib.util
    
    # Save current directory
    orig_dir = os.getcwd()
    
    try:
        os.chdir(setup_dir)
        
        # Load setup.py as a module
        spec = importlib.util.spec_from_file_location("setup_module", "setup.py")
        setup_module = importlib.util.module_from_spec(spec)
        
        # Mock the setup function to capture its arguments
        captured_args = {}
        
        def mock_setup(**kwargs):
            captured_args.update(kwargs)
        
        # Replace setuptools.setup with our mock
        import setuptools
        original_setup = setuptools.setup
        setuptools.setup = mock_setup
        
        # Execute the setup.py
        spec.loader.exec_module(setup_module)
        
        # Restore original setup
        setuptools.setup = original_setup
        
        return captured_args.get('install_requires', [])
    finally:
        os.chdir(orig_dir)


def test_standalone_dependencies():
    """Test that standalone mode includes config_explorer from git"""
    print("\nTest 1: Standalone mode dependencies")
    print("-" * 50)
    
    setup_dir = Path(__file__).parent.parent
    install_requires = get_install_requires(setup_dir)
    
    print("Install requires:")
    for dep in install_requires:
        print(f"  - {dep}")
    
    # Check that config_explorer is installed from git
    has_config_explorer = any('config_explorer' in dep for dep in install_requires)
    has_git_url = any(f'git+{LLM_D_BENCHMARK_REPO}' in dep for dep in install_requires)
    
    assert has_config_explorer, "Should include config_explorer in standalone mode"
    assert has_git_url, "Should install config_explorer from git in standalone mode"
    print("✓ Standalone mode correctly includes config_explorer from git")
    return True


def test_submodule_dependencies():
    """Test that submodule mode excludes config_explorer"""
    print("\nTest 2: Submodule mode dependencies")
    print("-" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create the directory structure
        config_explorer_dir = tmpdir / "config_explorer"
        config_explorer_dir.mkdir()
        (config_explorer_dir / "pyproject.toml").write_text("[project]\nname = 'config_explorer'\n")
        
        config_recommender_dir = tmpdir / "config-recommender"
        config_recommender_dir.mkdir()
        
        # Copy necessary files
        original_dir = Path(__file__).parent.parent
        shutil.copy(original_dir / "setup.py", config_recommender_dir / "setup.py")
        shutil.copy(original_dir / "README.md", config_recommender_dir / "README.md")
        
        # Get dependencies
        install_requires = get_install_requires(config_recommender_dir)
        
        print("Install requires:")
        for dep in install_requires:
            print(f"  - {dep}")
        
        # Check that config_explorer is NOT in install_requires
        has_config_explorer = any('config_explorer' in dep for dep in install_requires)
        
        assert not has_config_explorer, "Should NOT include config_explorer in submodule mode"
        
        # But other dependencies should still be there
        has_llm_optimizer = any('llm-optimizer' in dep for dep in install_requires)
        has_streamlit = any('streamlit' in dep for dep in install_requires)
        has_pandas = any('pandas' in dep for dep in install_requires)
        
        assert has_llm_optimizer, "Should still include llm-optimizer"
        assert has_streamlit, "Should still include streamlit"
        assert has_pandas, "Should still include pandas"
        
        print("✓ Submodule mode correctly excludes config_explorer")
        print("✓ Other dependencies are still present")
        return True


def test_env_variable_dependencies():
    """Test that env variable forces submodule mode"""
    print("\nTest 3: Environment variable override")
    print("-" * 50)
    
    # Set environment variable
    os.environ["CONFIG_RECOMMENDER_SUBMODULE_MODE"] = "1"
    
    try:
        setup_dir = Path(__file__).parent.parent
        install_requires = get_install_requires(setup_dir)
        
        print("Install requires (with CONFIG_RECOMMENDER_SUBMODULE_MODE=1):")
        for dep in install_requires:
            print(f"  - {dep}")
        
        # Check that config_explorer is NOT in install_requires
        has_config_explorer = any('config_explorer' in dep for dep in install_requires)
        
        assert not has_config_explorer, "Environment variable should force submodule mode (exclude config_explorer)"
        print("✓ Environment variable correctly forces submodule mode")
        return True
    finally:
        del os.environ["CONFIG_RECOMMENDER_SUBMODULE_MODE"]


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Setup.py Dependency Configuration")
    print("=" * 60)
    
    tests = [
        test_standalone_dependencies,
        test_submodule_dependencies,
        test_env_variable_dependencies,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

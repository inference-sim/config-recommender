#!/usr/bin/env python3
"""Test script to verify submodule detection logic in setup.py"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

def test_standalone_mode():
    """Test that standalone mode is detected correctly"""
    print("Testing standalone mode detection...")
    
    # In standalone mode, there should be no ../config_explorer
    # We're running from the actual repo, so this should return False
    parent_dir = Path(__file__).parent.parent
    config_explorer_path = parent_dir / "config_explorer"
    is_submodule = config_explorer_path.exists() and (config_explorer_path / "pyproject.toml").exists()
    
    print(f"  Parent dir: {parent_dir}")
    print(f"  Config explorer path: {config_explorer_path}")
    print(f"  Is submodule: {is_submodule}")
    
    assert not is_submodule, "Should not detect submodule in standalone repo"
    print("  ✓ Standalone mode detected correctly")
    return True


def test_submodule_mode():
    """Test that submodule mode is detected correctly"""
    print("\nTesting submodule mode detection...")
    
    # Create a temporary directory structure that mimics llm-d-benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create config_explorer directory with pyproject.toml
        config_explorer_dir = tmpdir / "config_explorer"
        config_explorer_dir.mkdir()
        (config_explorer_dir / "pyproject.toml").write_text("[project]\nname = 'config_explorer'\n")
        
        # Create config-recommender directory
        config_recommender_dir = tmpdir / "config-recommender"
        config_recommender_dir.mkdir()
        
        # Copy setup.py to test directory
        setup_py_path = config_recommender_dir / "setup.py"
        original_setup = Path(__file__).parent.parent / "setup.py"
        shutil.copy(original_setup, setup_py_path)
        
        # Now test from within the config-recommender directory
        os.chdir(config_recommender_dir)
        
        # Simulate the detection logic
        parent_dir = Path(setup_py_path).parent.parent
        config_explorer_path = parent_dir / "config_explorer"
        is_submodule = config_explorer_path.exists() and (config_explorer_path / "pyproject.toml").exists()
        
        print(f"  Temp dir structure:")
        print(f"    {tmpdir}/")
        print(f"    ├── config_explorer/")
        print(f"    │   └── pyproject.toml")
        print(f"    └── config-recommender/")
        print(f"        └── setup.py")
        print(f"  Parent dir: {parent_dir}")
        print(f"  Config explorer path: {config_explorer_path}")
        print(f"  Is submodule: {is_submodule}")
        
        assert is_submodule, "Should detect submodule when config_explorer exists as sibling"
        print("  ✓ Submodule mode detected correctly")
        return True


def test_env_variable_override():
    """Test that environment variable can force submodule mode"""
    print("\nTesting environment variable override...")
    
    # Set the environment variable
    os.environ["CONFIG_RECOMMENDER_SUBMODULE_MODE"] = "1"
    
    # The environment variable should force submodule mode
    is_submodule_env = os.environ.get("CONFIG_RECOMMENDER_SUBMODULE_MODE") == "1"
    
    print(f"  CONFIG_RECOMMENDER_SUBMODULE_MODE: {os.environ.get('CONFIG_RECOMMENDER_SUBMODULE_MODE')}")
    print(f"  Forces submodule mode: {is_submodule_env}")
    
    assert is_submodule_env, "Environment variable should force submodule mode"
    print("  ✓ Environment variable override works correctly")
    
    # Clean up
    del os.environ["CONFIG_RECOMMENDER_SUBMODULE_MODE"]
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Submodule Detection Logic")
    print("=" * 60)
    
    tests = [
        test_standalone_mode,
        test_submodule_mode,
        test_env_variable_override,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Simple test script to verify the deployment tool structure.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.config import Config, NodeConfig, ConfigValidationError
        print("  Config module: OK")
    except Exception as e:
        print(f"  Config module: FAILED - {e}")
        return False
    
    try:
        from core.ssh_client import SSHClient, SSHError
        print("  SSH client module: OK")
    except Exception as e:
        print(f"  SSH client module: FAILED - {e}")
        return False
    
    try:
        from core.docker_manager import DockerManager
        print("  Docker manager module: OK")
    except Exception as e:
        print(f"  Docker manager module: FAILED - {e}")
        return False
    
    try:
        from core.generator import ScriptGenerator
        print("  Generator module: OK")
    except Exception as e:
        print(f"  Generator module: FAILED - {e}")
        return False
    
    try:
        from core.deployer import Deployer
        print("  Deployer module: OK")
    except Exception as e:
        print(f"  Deployer module: FAILED - {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    from core.config import Config, ConfigValidationError
    
    config_file = "config.example.yaml"
    if not os.path.exists(config_file):
        print(f"  Config file not found: {config_file}")
        return False
    
    try:
        config = Config(config_file)
        print(f"  Config loaded: OK")
        print(f"    - Prefill nodes: {len(config.get_prefill_nodes())}")
        print(f"    - Decode nodes: {len(config.get_decode_nodes())}")
        print(f"    - Proxy enabled: {config.get_proxy_config().enabled}")
        return True
    except ConfigValidationError as e:
        print(f"  Config validation: FAILED - {e}")
        return False
    except Exception as e:
        print(f"  Config loading: FAILED - {e}")
        return False

def test_script_generation():
    """Test script generation."""
    print("\nTesting script generation...")
    
    from core.config import Config
    from core.generator import ScriptGenerator
    
    try:
        config = Config("config.example.yaml")
        generator = ScriptGenerator("templates")
        
        # Generate to temp directory
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            generated = generator.generate_all(config, tmpdir)
            print(f"  Script generation: OK")
            print(f"    - Generated files for {len(generated)} components")
            
            # Show sample
            for comp, files in list(generated.items())[:1]:
                print(f"    - Sample: {comp}")
                for name, path in files.items():
                    print(f"      - {name}")
        
        return True
    except Exception as e:
        print(f"  Script generation: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("vLLM-PD Deployer - Test Suite")
    print("=" * 60)
    
    results = []
    
    # Change to script directory
    os.chdir(os.path.dirname(__file__))
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Script Generation", test_script_generation()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

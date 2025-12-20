#!/usr/bin/env python3
"""
Diagnostic script to identify import issues with agent5 V4 pipeline.
Run this script to see exactly what's wrong.
"""

import sys
import os
from pathlib import Path

print("=" * 70)
print("Agent5 V4 Import Diagnostic Tool")
print("=" * 70)
print()

# 1. Python version
print(f"[1] Python Version: {sys.version}")
print(f"    Executable: {sys.executable}")
print()

# 2. Current directory
print(f"[2] Current Directory: {os.getcwd()}")
print()

# 3. Check if agent5 directory exists
agent5_path = Path("agent5")
if agent5_path.exists():
    print(f"[3] ✓ agent5 directory found")
else:
    print(f"[3] ✗ agent5 directory NOT found")
    print("    ERROR: You must run this from the Agent5 project root!")
    sys.exit(1)
print()

# 4. Check if bottom_up_aggregator.py exists
bottom_up_file = agent5_path / "bottom_up_aggregator.py"
if bottom_up_file.exists():
    print(f"[4] ✓ agent5/bottom_up_aggregator.py exists")
    
    # Check file size
    size = bottom_up_file.stat().st_size
    print(f"    File size: {size} bytes")
    
    # Check if function is defined
    with open(bottom_up_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'def aggregate_semantics' in content:
            print(f"    ✓ 'def aggregate_semantics' found in file")
            # Find line number
            for i, line in enumerate(content.split('\n'), 1):
                if line.startswith('def aggregate_semantics'):
                    print(f"    ✓ Function defined at line {i}")
                    break
        else:
            print(f"    ✗ 'def aggregate_semantics' NOT found in file!")
else:
    print(f"[4] ✗ agent5/bottom_up_aggregator.py NOT found")
    sys.exit(1)
print()

# 5. Check dependencies
print("[5] Checking Dependencies...")
required_packages = [
    'langchain_community',
    'langchain_core',
    'langchain_ollama',
]

missing = []
for package in required_packages:
    try:
        __import__(package)
        print(f"    ✓ {package} installed")
    except ImportError as e:
        print(f"    ✗ {package} NOT installed: {e}")
        missing.append(package)
print()

if missing:
    print(f"[!] ERROR: Missing dependencies: {', '.join(missing)}")
    print(f"    Run: pip install -r requirements.txt")
    print()

# 6. Try importing the module
print("[6] Testing Module Import...")
try:
    # Add current directory to path
    sys.path.insert(0, os.getcwd())
    
    import agent5.bottom_up_aggregator as ba
    print("    ✓ agent5.bottom_up_aggregator imported successfully!")
    
    # Check for function
    if hasattr(ba, 'aggregate_semantics'):
        print("    ✓ aggregate_semantics function is available!")
        print(f"    Function signature: {ba.aggregate_semantics.__name__}")
    else:
        print("    ✗ aggregate_semantics function NOT found in module!")
        print(f"    Available functions: {[x for x in dir(ba) if not x.startswith('_')]}")
        
except ImportError as e:
    print(f"    ✗ IMPORT ERROR: {e}")
    print()
    print("    Full traceback:")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"    ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
print()

# 7. Check other wrapper functions
print("[7] Checking Other Wrapper Functions...")
modules_to_check = [
    ('agent5.leaf_semantic_extractor', 'extract_leaf_semantics'),
    ('agent5.sfm_builder', 'build_scenario_flow_model'),
    ('agent5.mermaid_translator', 'translate_to_mermaid'),
]

for module_name, func_name in modules_to_check:
    try:
        module = __import__(module_name, fromlist=[func_name])
        if hasattr(module, func_name):
            print(f"    ✓ {module_name}.{func_name} available")
        else:
            print(f"    ✗ {module_name}.{func_name} NOT found")
    except ImportError as e:
        print(f"    ✗ {module_name} import failed: {e}")
print()

# 8. Check if installed as package
print("[8] Checking Package Installation...")
try:
    import pkg_resources
    try:
        version = pkg_resources.get_distribution('agent5').version
        location = pkg_resources.get_distribution('agent5').location
        print(f"    ✓ agent5 package installed")
        print(f"    Version: {version}")
        print(f"    Location: {location}")
    except pkg_resources.DistributionNotFound:
        print(f"    ✗ agent5 NOT installed as a package")
        print(f"    Run: pip install -e .")
except ImportError:
    print(f"    ! Cannot check (pkg_resources not available)")
print()

print("=" * 70)
print("Diagnosis Complete")
print("=" * 70)
print()
print("If all checks pass but you still get import errors:")
print("1. Clear Python cache: rm -rf **/__pycache__ **/*.pyc")
print("2. Reinstall: pip uninstall agent5 -y && pip install -e .")
print("3. Check you're using the correct Python: which python3")
print()





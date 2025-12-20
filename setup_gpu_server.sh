#!/bin/bash
# Agent5 V4 - GPU Server Setup Script
# Run this on your GPU server after cloning

set -e  # Exit on error

echo "=========================================="
echo "Agent5 V4 - GPU Server Setup"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    echo "Please cd to the Agent5 directory first"
    exit 1
fi

# Step 1: Checkout version4 branch
echo "[1/5] Checking out version4 branch..."
git checkout version4
git pull origin version4
echo "✓ On version4 branch"
echo ""

# Step 2: Create virtual environment (optional but recommended)
echo "[2/5] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Step 3: Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Step 4: Install dependencies
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 5: Install agent5
echo "[5/5] Installing agent5 in editable mode..."
pip install -e .
echo "✓ Agent5 installed"
echo ""

# Verify installation
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
echo ""

# Check commands
if command -v agent5 &> /dev/null; then
    echo "✓ agent5 command available"
else
    echo "✗ agent5 command NOT found"
fi

if command -v agent5-v4 &> /dev/null; then
    echo "✓ agent5-v4 command available"
else
    echo "✗ agent5-v4 command NOT found"
fi

# Check module imports
python3 -c "
import agent5.bottom_up_aggregator as ba
print('✓ bottom_up_aggregator imports successfully')
print(f'✓ aggregate_semantics function exists: {hasattr(ba, \"aggregate_semantics\")}')

import agent5.leaf_semantic_extractor as lse
print(f'✓ extract_leaf_semantics function exists: {hasattr(lse, \"extract_leaf_semantics\")}')

import agent5.sfm_builder as sb
print(f'✓ build_scenario_flow_model function exists: {hasattr(sb, \"build_scenario_flow_model\")}')

import agent5.mermaid_translator as mt
print(f'✓ translate_to_mermaid function exists: {hasattr(mt, \"translate_to_mermaid\")}')
" 2>&1

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Test the V4 command:"
echo "  agent5-v4 --help"
echo ""
echo "Run a flowchart generation:"
echo "  agent5-v4 \\"
echo "    --project-path ./your_cpp_project \\"
echo "    --entry-function main \\"
echo "    --detail-level medium \\"
echo "    --out flowchart.mmd"
echo ""





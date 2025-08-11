#!/bin/bash
"""
Setup script to create and configure a virtual environment for the gentv project.
This ensures all dependencies are isolated and the project is properly containerized.
"""

set -e  # Exit on any error

echo "🐍 Setting up virtual environment for gentv project"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    exit 1
fi

python_version=$(python3 --version)
echo "✓ Found $python_version"

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf venv
fi

# Create new virtual environment
echo "🔨 Creating new virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install project in development mode
echo "📦 Installing gentv package in development mode..."
pip install -e .

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install pytest pytest-cov python-dotenv

# Install LLM provider dependencies
echo "📦 Installing LLM provider dependencies..."
pip install openai anthropic

echo ""
echo "✅ Virtual environment setup complete!"
echo ""
echo "🚀 To activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "🧪 To run tests:"
echo "   pytest tests/"
echo ""
echo "🎯 To run integration tests:"
echo "   python run_integration_tests.py --quick"
echo ""
echo "📋 To deactivate when done:"
echo "   deactivate"
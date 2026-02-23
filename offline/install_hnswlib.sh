#!/bin/bash
# Script to install hnswlib and its dependencies
# This handles the compilation requirements for hnswlib on Ubuntu/Debian systems

set -e  # Exit on error

echo "Installing hnswlib dependencies and package..."

# Check if running on Ubuntu/Debian
if ! command -v apt &> /dev/null; then
    echo "Warning: This script is designed for Ubuntu/Debian systems with apt package manager"
fi

# Install C++ compiler if not present
if ! command -v g++ &> /dev/null; then
    echo "Installing g++ compiler..."
    sudo apt update
    sudo apt install -y g++
else
    echo "g++ compiler already installed"
fi

# Install Python development headers
PYTHON_VERSION=$(python3 --version | grep -oP '3\.\d+')
echo "Detected Python version: $PYTHON_VERSION"

if ! dpkg -l | grep -q "python${PYTHON_VERSION}-dev"; then
    echo "Installing Python development headers..."
    sudo apt install -y python${PYTHON_VERSION}-dev
else
    echo "Python development headers already installed"
fi

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No .venv directory found. Installing to system Python."
fi

# Install pybind11 (required by hnswlib)
echo "Installing pybind11..."
pip install pybind11

# Install hnswlib
echo "Installing hnswlib..."
pip install hnswlib

echo ""
echo "✓ hnswlib installation completed successfully!"
echo ""
echo "Installed versions:"
pip show hnswlib | grep -E "Name|Version"

#!/bin/bash

# Setup script for the pipeline

echo "================================"
echo "Setting up iREL_TASK Pipeline"
echo "================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Download spaCy models
echo ""
echo "Downloading spaCy models..."
python3 -m spacy download xx_ent_wiki_sm

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/output
mkdir -p data/output/visualizations
mkdir -p logs

echo ""
echo "================================"
echo "Setup completed successfully!"
echo "================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python pipeline.py"
echo ""
echo "Next steps:"
echo "1. Add YouTube video URLs to config/videos.yaml"
echo "2. Run: python pipeline.py"

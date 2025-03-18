#!/bin/bash

# Set up virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo "Run 'source venv/bin/activate' to activate the environment."

#!/bin/bash

# Variables
VENV_DIR=.venv

echo "Removing virtual environment: $VENV_DIR/"
rm -rf $VENV_DIR
echo "Removing all __pycache__ folders..."
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "Cleanup complete."
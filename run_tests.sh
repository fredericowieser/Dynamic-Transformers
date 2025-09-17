#!/bin/bash

echo "Running Dynamic Transformer Test Suite"
echo "======================================="
echo ""

# Run unit tests for layers
echo "Running layer tests..."
uv run pytest tests/test_layers.py -v --tb=short -q

# Run unit tests for routers
echo ""
echo "Running router tests..."
uv run pytest tests/test_routers.py -v --tb=short -q

# Run unit tests for blocks
echo ""
echo "Running block tests..."
uv run pytest tests/test_blocks.py -v --tb=short -q

# Run unit tests for models
echo ""
echo "Running model tests..."
uv run pytest tests/test_models.py -v --tb=short -q

# Run integration tests for training
echo ""
echo "Running training integration tests..."
uv run pytest tests/test_training_integration.py::TestTrainingIntegration::test_training_step_vpr -v --tb=short -q

# Run integration tests for evaluation
echo ""
echo "Running evaluation integration tests..."
uv run pytest tests/test_evaluation_integration.py::TestEvaluationIntegration::test_model_generation_evaluation -v --tb=short -q

echo ""
echo "======================================="
echo "Test suite complete!"
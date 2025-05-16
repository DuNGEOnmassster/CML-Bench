#!/bin/bash

# Script to run benchmark_metrics_v4.py with specified GPU IDs
# Usage: ./run_benchmark_v4.sh [GPU_IDs]
# Example: ./run_benchmark_v4.sh 0,1,2,3
# If no GPU_IDs are provided, CUDA_VISIBLE_DEVICES will not be explicitly set,
# and PyTorch/Transformers will use its default behavior.

# Check if GPU IDs are provided as the first argument
if [ -n "$1" ]; then
  GPU_IDS="$1"
  echo "Using specified GPU IDs: $GPU_IDS"
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
else
  echo "No GPU IDs provided. CUDA_VISIBLE_DEVICES will be default to all available GPUs (PyTorch default)."
  # Ensure CUDA_VISIBLE_DEVICES is unset if no argument is given, to allow PyTorch default.
  # If you always want to restrict to certain GPUs by default, set it here, e.g.:
  # export CUDA_VISIBLE_DEVICES="0" 
fi

# Get the directory where the shell script is located
# This ensures that the script can be run from any directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define the path to the Python script
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/scripts/benchmark_metrics_v5.py"

# Define the results directory and log file path
RESULTS_SUBDIR="0515_final" # Match the subdir in the python script
# INPUT_ROOT="$SCRIPT_DIR/data/ground_truth/gt_100.json, $SCRIPT_DIR/data/result/gt_100"
INPUT_ROOT="$SCRIPT_DIR/CML-Dataset/instruction, $SCRIPT_DIR/CML-Dataset/base"
RESULTS_DIR="$SCRIPT_DIR/scripts/results/${RESULTS_SUBDIR}"
LOG_FILE="$RESULTS_DIR/output_benchmark_v5_0515_final.log" # Match the log file convention

# Create the results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
  echo "Error: Python script not found at $PYTHON_SCRIPT_PATH" | tee -a "$LOG_FILE"
  exit 1
fi

echo "Running Python script: $PYTHON_SCRIPT_PATH"
echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES (If blank, PyTorch default)"
echo "Logging output to: $LOG_FILE"

# Execute the Python script and redirect stdout and stderr to the log file
# Using ">" to overwrite for a fresh log for each run. Use ">>" to append.
python -u "$PYTHON_SCRIPT_PATH" --input_root "$INPUT_ROOT" --results_dir "$RESULTS_DIR" > "$LOG_FILE" 2>&1

# Check the exit status of the Python script
EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]; then
  echo "Python script exited with error code $EXIT_STATUS. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
else
  echo "Benchmark script finished successfully. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
fi

exit $EXIT_STATUS

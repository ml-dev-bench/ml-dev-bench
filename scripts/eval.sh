#!/bin/sh

# Usage: ./eval.sh <task_config> [agent_config] [additional_hydra_args]
# Example: ./eval.sh task=hello_world agent=openhands
# Example with overrides: ./eval.sh task=hello_world agent=openhands num_runs=3

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Navigate to the project root (ml_dev_bench)
cd "${SCRIPT_DIR}/.." || exit 1

# Run the evaluation using Hydra through Poetry
python -m calipers.scripts.run_hydra_evaluation $@

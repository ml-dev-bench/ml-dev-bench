#!/bin/sh

# Usage: ./eval.sh <config_path> <tasks> <commit_hash>
# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Parse arguments
CONFIG_PATH=$1
TASKS=$2
COMMIT_HASH=$3

# Set up optional arguments
TASKS_ARG=""
if [ ! -z "$TASKS" ]; then
    TASKS_ARG="--tasks $TASKS"
fi

COMMIT_ARG=""
if [ ! -z "$COMMIT_HASH" ]; then
    COMMIT_ARG="--commit $COMMIT_HASH"
fi

# Navigate to the project root (ml_dev_bench)
cd "${SCRIPT_DIR}/.." || exit 1

# Run the evaluation using the provided config and optional arguments
python -m calipers.scripts.run_evaluation \
    --config $CONFIG_PATH \
    $TASKS_ARG \
    $COMMIT_ARG

# OpenHands Agent

The OpenHands agent is a powerful evaluation agent that uses the OpenHands framework to run evaluations in a sandboxed Docker environment.

## Prerequisites

1. Docker Desktop installed and running
2. Build the runtime image from ml-dev-bench fork:
   ```bash
   # Clone the repository
   git clone https://github.com/ml-dev-bench/OpenHands.git
   cd OpenHands
   git checkout ml-dev-bench-v0.21.1

   # Build the runtime image
   # TODO: add instructions here
   ```
3. Host network feature enabled in Docker Desktop (required for MacOS)

## Running Evaluations

### Basic Usage

```bash
# Run a single evaluation
./scripts/eval.sh <path_to_config_yaml>

# Example with hello world task
./scripts/eval.sh ml_dev_bench/cases/hello_world/hello_world_config.yaml
```

### Results

Evaluation results will be saved to:
- JSON format: `./results/eval_results_<timestamp>.json`
- Contains metrics, success rate, and execution history

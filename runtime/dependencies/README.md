# ML-Dev-Bench Runtime Dependencies

This directory contains the dependencies required for the ML-Dev-Bench runtime environment. The runtime environment is used to evaluate the AI agent generated code in isolated environments.

## Overview

The runtime dependencies are managed separately from the main project dependencies to:
1. Keep the core evaluation framework lightweight
2. Enable isolated execution environments

## Installation

The runtime dependencies are installed automatically when you run:

```bash
make install-runtime-dependencies
```

This command will:
1. Create a separate Poetry environment for runtime dependencies
2. Install all required packages for task execution
3. Configure the runtime environment

## Development

If you need to modify the runtime dependencies:

1. Update the `pyproject.toml` in this directory
2. Run `make install-runtime-dependencies` from the project root
3. Test your changes using the framework's test suite

## Note

The runtime dependencies are essential for local task execution. Make sure to install them before running evaluations locally.

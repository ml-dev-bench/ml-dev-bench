# ML-Dev-Bench

ML-Dev-Bench is an evaluation bench for evaluating AI agents against various ML development tasks.

Calipers is a framework for evaluating AI agents, providing tools and infrastructure for systematic assessment of AI model performance.

ML-Dev-Bench features:
- Flexible evaluation framework for AI agents
- Support for multiple runtime environments (Local, Docker, E2B)
- Comprehensive metrics tracking and reporting
- Integration with LiteLLM and LangChain
- Configurable task-based evaluation system

## Features


## Requirements

- Python 3.12+
- Poetry 1.8+
- Linux, macOS, or Windows Subsystem for Linux (WSL)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ml-dev-bench/ml-dev-bench.git
cd ml-dev-bench
```

2. Install dependencies:

```bash
make build
```
This will:
- Check system requirements
- Install Python dependencies
- Set up pre-commit hooks
- Configure the development environment

3. Install runtime dependencies:

This is needed for running evaluations locally.

```bash
make install-runtime-dependencies
```

## Usage

1. Run evaluations using the command-line interface:

```bash
./scripts/eval.sh --config path/to/eval_config.yaml

```

To run the hello world evaluation:

```bash
./scripts/eval.sh --config ml_dev_bench/cases/hello_world/hello_world_config.yaml
```

## Development

- Format and lint code:

```bash
make lint
```

## Adding New Agent Groups

To add a new agent group with its specific dependencies:

1. Add a new group in `pyproject.toml`:
```toml
[tool.poetry.group.{your-agent-name}.dependencies]
dependency1 = "^version"
dependency2 = "^version"
```

2. Add a corresponding make target in `Makefile`:
```makefile
install-{your-agent}-dependencies:
	@echo "$(GREEN)Installing Python dependencies with {your-agent} in new environment...$(RESET)"
	POETRY_VIRTUALENVS_PATH="./.venv-{your-agent}" poetry env use python$(PYTHON_VERSION)
	POETRY_VIRTUALENVS_PATH="./.venv-{your-agent}" poetry install --with {your-agent}
```

This creates a separate virtual environment with a suffix matching your agent name (e.g., `.venv-{your-agent}`).

Example: The react-agent group is set up with:
```bash
make install-react-agent-dependencies
```
This creates a dedicated environment at `.venv-react` with all react-agent specific dependencies.

## Project Structure

```
.
├── calipers/
│   ├── agents/          # Agent implementations
│   ├── callbacks/       # Callback handlers
│   ├── framework/       # Core evaluation framework
│   ├── metrics/         # Metrics tracking
│   └── scripts/         # CLI tools
│
└── runtime/
    ├── backends/        # Runtime backend implementations
    ├── environments/    # Environment configurations
    └── tools/           # Runtime tools
```

## Adding New Agents

### Directory Structure
To add a new agent:
1. Create a new directory under `agents/` with your agent name (e.g., `agents/my_agent/`)
2. Add your agent implementation files in this directory
3. Create a `Dockerfile` in your agent directory that extends the base image

Example structure:
```
agents/
├── env/
│   └── base.Dockerfile    # Base Docker image
├── my_agent/
│   ├── __init__.py
│   ├── my_agent.py       # Your agent implementation
│   └── Dockerfile        # Agent-specific Dockerfile
└── utils.py              # Shared utilities
```

### Docker Setup
The project uses a two-stage Docker build:
1. A base image with core dependencies
2. Agent-specific images that extend the base image

#### Building Images
1. Build the base image (from project root):
```bash
docker build -t ml-dev-bench-base -f env/base.Dockerfile .
```

2. Build your agent's image (from project root):
```bash
docker build -t ml-dev-bench-myagent -f agents/my_agent/Dockerfile .
```

#### Creating Agent Dockerfile
Your agent's Dockerfile should:
1. Extend the base image
2. Copy agent-specific code
3. Install agent-specific dependencies

Example agent Dockerfile:
```dockerfile
FROM ml-dev-bench-base:latest

# Copy the agent code
COPY agents/my_agent/ ./agents/my_agent/
COPY agents/__init__.py ./agents/
COPY agents/utils.py ./agents/

# Install agent-specific dependencies
RUN poetry install --with my-agent

# Set working directory
WORKDIR $WORKDIR/agents/my_agent

# Default command - open a shell with poetry env
CMD ["poetry", "shell"]
```
## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run linters and tests
5. Submit a pull request

## License

MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- LiteLLM for LLM integration
- Composio for runtime management

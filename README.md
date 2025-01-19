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

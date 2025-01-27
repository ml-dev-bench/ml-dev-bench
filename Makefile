SHELL=/bin/bash
# Makefile for project

# Variables
DEFAULT_WORKSPACE_DIR = "./workspace"
DEFAULT_MODEL = "gpt-4o"
CONFIG_FILE = config.toml
PRE_COMMIT_CONFIG_PATH = "./dev_config/python/.pre-commit-config.yaml"
PYTHON_VERSION = 3.12

# ANSI color codes
GREEN=$(shell tput -Txterm setaf 2)
YELLOW=$(shell tput -Txterm setaf 3)
RED=$(shell tput -Txterm setaf 1)
BLUE=$(shell tput -Txterm setaf 6)
RESET=$(shell tput -Txterm sgr0)

# Build
deploy-build:
	@echo "$(GREEN)Building project...$(RESET)"
	@$(MAKE) -s check-dependencies
	@$(MAKE) -s install-deploy-python-dependencies
	@echo "$(GREEN)Build completed successfully.$(RESET)"

build:
	@echo "$(GREEN)Building project...$(RESET)"
	@$(MAKE) -s check-dependencies
	@$(MAKE) -s install-python-dependencies
	@$(MAKE) -s install-pre-commit-hooks
	@echo "$(GREEN)Build completed successfully.$(RESET)"

check-dependencies:
	@echo "$(YELLOW)Checking dependencies...$(RESET)"
	@$(MAKE) -s check-system
	@$(MAKE) -s check-python
	@$(MAKE) -s check-poetry
	@echo "$(GREEN)Dependencies checked successfully.$(RESET)"

check-system:
	@echo "$(YELLOW)Checking system...$(RESET)"
	@if [ "$(shell uname)" = "Darwin" ]; then \
		echo "$(BLUE)macOS detected.$(RESET)"; \
	elif [ "$(shell uname)" = "Linux" ]; then \
		if [ -f "/etc/manjaro-release" ]; then \
			echo "$(BLUE)Manjaro Linux detected.$(RESET)"; \
		else \
			echo "$(BLUE)Linux detected.$(RESET)"; \
		fi; \
	elif [ "$$(uname -r | grep -i microsoft)" ]; then \
		echo "$(BLUE)Windows Subsystem for Linux detected.$(RESET)"; \
	else \
		echo "$(RED)Unsupported system detected. Please use macOS, Linux, or Windows Subsystem for Linux (WSL).$(RESET)"; \
		exit 1; \
	fi

check-python:
	@echo "$(YELLOW)Checking Python installation...$(RESET)"
	@if command -v python$(PYTHON_VERSION) > /dev/null; then \
		echo "$(BLUE)$(shell python$(PYTHON_VERSION) --version) is already installed.$(RESET)"; \
	else \
		echo "$(RED)Python $(PYTHON_VERSION) is not installed. Please install Python $(PYTHON_VERSION) to continue.$(RESET)"; \
		exit 1; \
	fi

check-poetry:
	@echo "$(YELLOW)Checking Poetry installation...$(RESET)"
	@if command -v poetry > /dev/null; then \
		POETRY_VERSION=$(shell poetry --version 2>&1 | sed -E 's/Poetry \(version ([0-9]+\.[0-9]+\.[0-9]+)\)/\1/'); \
		IFS='.' read -r -a POETRY_VERSION_ARRAY <<< "$$POETRY_VERSION"; \
		if [ $${POETRY_VERSION_ARRAY[0]} -ge 1 ] && [ $${POETRY_VERSION_ARRAY[1]} -ge 8 ]; then \
			echo "$(BLUE)$(shell poetry --version) is already installed.$(RESET)"; \
		else \
			echo "$(RED)Poetry 1.8 or later is required. You can install poetry by running the following command, then adding Poetry to your PATH:"; \
			echo "$(RED) curl -sSL https://install.python-poetry.org | python$(PYTHON_VERSION) -$(RESET)"; \
			echo "$(RED)More detail here: https://python-poetry.org/docs/#installing-with-the-official-installer$(RESET)"; \
			exit 1; \
		fi; \
	else \
		echo "$(RED)Poetry is not installed. You can install poetry by running the following command, then adding Poetry to your PATH:"; \
		echo "$(RED) curl -sSL https://install.python-poetry.org | python$(PYTHON_VERSION) -$(RESET)"; \
		echo "$(RED)More detail here: https://python-poetry.org/docs/#installing-with-the-official-installer$(RESET)"; \
		exit 1; \
	fi

install-deploy-python-dependencies:
	@echo "$(GREEN)Installing Python dependencies...$(RESET)"
	@if [ -z "${TZ}" ]; then \
		echo "Defaulting TZ (timezone) to UTC"; \
		export TZ="UTC"; \
	fi
	poetry env use python$(PYTHON_VERSION)
	@poetry install --without dev,test
	@echo "$(GREEN)Python dependencies installed successfully.$(RESET)"

install-python-dependencies:
	@echo "$(GREEN)Installing Python dependencies...$(RESET)"
	@if [ -z "${TZ}" ]; then \
		echo "Defaulting TZ (timezone) to UTC"; \
		export TZ="UTC"; \
	fi
	poetry config installer.parallel true
	poetry config virtualenvs.create true
	poetry env use python$(PYTHON_VERSION)
	@POETRY_REQUESTS_TIMEOUT=120 poetry install --no-interaction --verbose
	@echo "$(GREEN)Python dependencies installed successfully.$(RESET)"

install-react-agent-dependencies:
	@echo "$(GREEN)Installing Python dependencies with react-agent in new environment...$(RESET)"
	@if [ -z "${TZ}" ]; then \
		echo "Defaulting TZ (timezone) to UTC"; \
		export TZ="UTC"; \
	fi
	@echo "$(YELLOW)Creating new virtual environment for react-agent...$(RESET)"
	POETRY_VIRTUALENVS_PATH="./.venv-react" poetry env use python$(PYTHON_VERSION)
	POETRY_VIRTUALENVS_PATH="./.venv-react" poetry install --with react-agent
	@echo "$(GREEN)Python dependencies with react-agent installed successfully in .venv-react$(RESET)"
	@echo "$(BLUE)To activate this environment, run: POETRY_VIRTUALENVS_PATH='./.venv-react' poetry shell$(RESET)"

install-aide-agent-dependencies:
	@echo "$(GREEN)Installing Python dependencies with aide-agent in new environment...$(RESET)"
	@if [ -z "${TZ}" ]; then \
		echo "Defaulting TZ (timezone) to UTC"; \
		export TZ="UTC"; \
	fi
	@echo "$(YELLOW)Creating new virtual environment for aide-agent...$(RESET)"
	@rm -rf .venv-aide
	@mkdir -p .venv-aide
	POETRY_VIRTUALENVS_CREATE=true \
	POETRY_VIRTUALENVS_IN_PROJECT=false \
	POETRY_VIRTUALENVS_PATH="$(PWD)/.venv-aide" \
	poetry env use python$(PYTHON_VERSION)
	POETRY_VIRTUALENVS_CREATE=true \
	POETRY_VIRTUALENVS_IN_PROJECT=false \
	POETRY_VIRTUALENVS_PATH="$(PWD)/.venv-aide" \
	poetry install --with aide
	@echo "$(GREEN)Python dependencies with aide-agent installed successfully in .venv-aide$(RESET)"
	@echo "$(BLUE)To activate this environment, run: POETRY_VIRTUALENVS_PATH='$(PWD)/.venv-aide' poetry shell$(RESET)"

install-runtime-dependencies:
	@echo "$(GREEN)Installing runtime Python dependencies...$(RESET)"
	@if [ -z "${TZ}" ]; then \
		echo "Defaulting TZ (timezone) to UTC"; \
		export TZ="UTC"; \
	fi
	cd runtime/dependencies && poetry env use python$(PYTHON_VERSION)
	@cd runtime/dependencies && poetry install --no-root
	@echo "$(GREEN)Runtime dependencies installed successfully.$(RESET)"

install-gcloud-key:
	@echo "$(YELLOW)Installing Google Cloud SDK repository key...$(RESET)"
	@sudo apt-get install apt-transport-https ca-certificates gnupg -y
	@sudo rm -f /usr/share/keyrings/cloud.google.gpg
	@curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
	@echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list
	@curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
	@sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C0BA5CE6DC6315A3
	@sudo apt-get update
	@echo "$(GREEN)Google Cloud SDK repository key installed successfully.$(RESET)"

install-openhands-agent-dependencies:
	@echo "$(GREEN)Installing Python dependencies with openhands-agent in new environment...$(RESET)"
	@if [ -z "${TZ}" ]; then \
		echo "Defaulting TZ (timezone) to UTC"; \
		export TZ="UTC"; \
	fi
	@echo "$(YELLOW)Creating new virtual environment for openhands-agent...$(RESET)"
	@rm -rf .venv-openhands
	@mkdir -p .venv-openhands
	POETRY_VIRTUALENVS_CREATE=true \
	POETRY_VIRTUALENVS_IN_PROJECT=false \
	POETRY_VIRTUALENVS_PATH="$(PWD)/.venv-openhands" \
	poetry env use python$(PYTHON_VERSION)
	POETRY_VIRTUALENVS_CREATE=true \
	POETRY_VIRTUALENVS_IN_PROJECT=false \
	POETRY_VIRTUALENVS_PATH="$(PWD)/.venv-openhands" \
	poetry install --with openhands-agent || ($(MAKE) -s install-gcloud-key && POETRY_VIRTUALENVS_PATH="$(PWD)/.venv-openhands" poetry install --with openhands-agent)
	POETRY_VIRTUALENVS_PATH="$(PWD)/.venv-openhands" \
	poetry run playwright install --with-deps chromium
	@echo "$(GREEN)Python dependencies with openhands-agent installed successfully in .venv-openhands$(RESET)"
	@echo "$(BLUE)To activate this environment, run: POETRY_VIRTUALENVS_PATH='$(PWD)/.venv-openhands' poetry shell$(RESET)"

install-pre-commit-hooks:
	@echo "$(YELLOW)Installing pre-commit hooks...$(RESET)"
	@git config --unset-all core.hooksPath || true
	@poetry run pre-commit install --config $(PRE_COMMIT_CONFIG_PATH)
	@echo "$(GREEN)Pre-commit hooks installed successfully.$(RESET)"

lint:
	@echo "$(YELLOW)Running linters...$(RESET)"
	@poetry run pre-commit run --files /**/* --show-diff-on-failure --config $(PRE_COMMIT_CONFIG_PATH)

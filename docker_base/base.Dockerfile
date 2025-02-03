FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    WORKDIR="/app"

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR $WORKDIR

# Copy only dependency files
COPY pyproject.toml poetry.lock ./

# Install main project dependencies
RUN poetry install --no-root --without dev

# Copy the core application code
COPY ml_dev_bench/ ./ml_dev_bench/
COPY calipers/ ./calipers/
COPY runtime/ ./runtime/
COPY README.md ./

# Install the project
RUN poetry install --only-root

# Set the default command
CMD ["poetry", "shell"]

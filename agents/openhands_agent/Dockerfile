# Start from our base image
FROM ml-dev-bench-base:latest

# Copy our agent code
COPY agents/openhands_agent/ ./openhands_agent/
COPY agents/__init__.py ./agents/
COPY agents/utils.py ./agents/

# Install dependencies
RUN poetry install --with openhands-agent
RUN poetry run playwright install --with-deps chromium

# Set working directory to our agent
WORKDIR $WORKDIR/agents/openhands_agent

# Default command
CMD ["poetry", "shell"]

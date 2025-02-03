FROM ml-dev-bench-base:latest

# Copy the AIDE agent code
COPY agents/aide_agent/ ./agents/aide_agent/
COPY agents/__init__.py ./agents/
COPY agents/utils.py ./agents/

# Install AIDE agent specific dependencies
RUN poetry install --with aide

# Set working directory to aide_agent
WORKDIR $WORKDIR/agents/aide_agent

# Default command - open a shell with poetry env
CMD ["poetry", "shell"]

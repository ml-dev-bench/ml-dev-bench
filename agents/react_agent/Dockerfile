FROM ml-dev-bench-base:latest

# Copy the react agent code
COPY agents/react_agent/ ./agents/react_agent/
COPY agents/__init__.py ./agents/
COPY agents/utils.py ./agents/

# Install react-agent specific dependencies
RUN poetry install --with react-agent

# Set working directory to react_agent
WORKDIR $WORKDIR/agents/react_agent

# Default command - open a shell with poetry env
CMD ["poetry", "shell"]

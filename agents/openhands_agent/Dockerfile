# Start from our base image
FROM ml-dev-bench-base:latest

# Set up environment variables for OpenHands
ARG LOGS_DIR=/home/logs
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR=/home/code
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR=/home/agent
ENV AGENT_DIR=${AGENT_DIR}
ARG OH_VERSION=0.20.0
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    netcat-openbsd \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p ${LOGS_DIR} ${AGENT_DIR} ${CODE_DIR} ${AGENT_DIR}/cache

# Copy our agent code first
COPY agents/openhands_agent/ ./openhands_agent/
COPY agents/__init__.py ./agents/
COPY agents/utils.py ./agents/

# Install our agent's dependencies first
RUN poetry config virtualenvs.create true && \
    poetry env use python3.12 && \
    poetry install --with openhands-agent

# Clone OpenHands
WORKDIR ${AGENT_DIR}
RUN git clone --branch ${OH_VERSION} --single-branch https://github.com/All-Hands-AI/OpenHands.git .

# Install OpenHands dependencies
RUN poetry install --without llama-index --no-root && \
    poetry run playwright install --with-deps chromium && \
    touch cache/playwright_chromium_is_installed.txt

# Set working directory to our agent
WORKDIR $WORKDIR/agents/openhands_agent

# Default command - open a shell with poetry env
CMD ["poetry", "shell"]

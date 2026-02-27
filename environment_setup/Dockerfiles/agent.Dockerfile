# Use stable Ubuntu LTS
FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    bash-completion \
    asciinema \
    ffmpeg \
    imagemagick \
    libagg-dev \
    ca-certificates \
    libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (recommended)
RUN useradd -m -d /home/useragent -s /bin/bash agentuser

# Set working directory
WORKDIR /home/useragent/codebase

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first (better layer caching)
COPY environment_setup/requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install Rust toolchain (nightly) for agg build
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
ENV PATH=/root/.cargo/bin:$PATH
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal \
    && rustup toolchain install nightly \
    && rustup default nightly

# Install asciinema-agg from source
RUN cargo install --git https://github.com/asciinema/agg \
    && mv /root/.cargo/bin/agg /usr/local/bin/agg

# Copy application code
COPY codebase/ /home/useragent/codebase/

# Enable bash completion for interactive shells
RUN echo 'if [ -f /etc/bash_completion ]; then . /etc/bash_completion; fi' >> /home/useragent/.bashrc \
    && chown -R agentuser:agentuser /home/useragent
USER agentuser

## Default command intentionally omitted; override in compose for dev

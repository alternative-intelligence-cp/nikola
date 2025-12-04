# APPENDIX G: DOCKER DEPLOYMENT SPECIFICATION

## G.1 Multi-Stage Dockerfile

**Status:** MANDATORY - Production deployment uses Docker containers

### G.1.1 Complete Dockerfile

**File:** `Dockerfile`

```dockerfile
# ============================================================================
# Stage 1: Build Environment
# ============================================================================
FROM ubuntu:24.04 AS builder

# Set non-interactive frontend
ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    liblmdb-dev \
    libvirt-dev \
    libcurl4-openssl-dev \
    libmagic-dev \
    libsodium-dev \
    libeigen3-dev \
    libfftw3-dev \
    libopencv-dev \
    nlohmann-json3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for GGUF export
RUN pip3 install --no-cache-dir gguf numpy

# Copy source code
WORKDIR /build
COPY . .

# Generate Protocol Buffer code
WORKDIR /build/proto
RUN protoc --cpp_out=../src/generated neural_spike.proto

# Build Nikola
WORKDIR /build
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-13 \
        -DENABLE_AVX512=ON \
        -DENABLE_CUDA=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_BENCHMARKS=OFF && \
    make -j$(nproc) && \
    make install DESTDIR=/install

# ============================================================================
# Stage 2: Runtime Environment
# ============================================================================
FROM ubuntu:24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies ONLY
RUN apt-get update && apt-get install -y \
    libzmq5 \
    libprotobuf32 \
    liblmdb0 \
    libvirt0 \
    libcurl4 \
    libmagic1 \
    libsodium23 \
    libfftw3-3 \
    libopencv-core4.6 \
    qemu-system-x86 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for runtime
RUN pip3 install --no-cache-dir gguf numpy

# Copy binaries and libraries from builder
COPY --from=builder /install/usr/local /usr/local

# Copy configuration files
COPY config/*.conf /etc/nikola/

# Create necessary directories
RUN mkdir -p \
    /var/lib/nikola/state \
    /var/lib/nikola/ingest \
    /var/lib/nikola/archive \
    /var/log/nikola \
    /tmp/nikola \
    && chmod 755 /tmp/nikola

# Set up permissions for KVM
RUN addgroup --gid 999 kvm || true && \
    usermod -aG kvm root

# Expose ZeroMQ Spine ports (if using TCP instead of IPC)
EXPOSE 5555 5556

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /usr/local/bin/twi-ctl status || exit 1

# Declare volumes for state persistence
# CurveZMQ keys and system state must persist across container restarts
VOLUME ["/var/lib/nikola/state", "/var/lib/nikola/ingest", "/var/lib/nikola/archive", "/etc/nikola/keys"]

# Default command: start daemon
ENTRYPOINT ["/usr/local/bin/nikola-daemon"]
CMD []
```

### G.1.2 CUDA-Enabled Dockerfile

**File:** `Dockerfile.cuda`

```dockerfile
# ============================================================================
# CUDA-Enabled Build (for GPU acceleration)
# ============================================================================
FROM nvidia/cuda:12.2.0-devel-ubuntu24.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies (same as standard Dockerfile)
RUN apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libzmq3-dev libprotobuf-dev protobuf-compiler \
    liblmdb-dev libvirt-dev libcurl4-openssl-dev \
    libsodium-dev libfftw3-dev libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# Build with CUDA support
RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DENABLE_CUDA=ON \
        -DENABLE_AVX512=ON \
        -DBUILD_TESTS=OFF && \
    make -j$(nproc) && \
    make install DESTDIR=/install

# ============================================================================
# Runtime with CUDA
# ============================================================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu24.04 AS runtime

RUN apt-get update && apt-get install -y \
    libzmq5 libprotobuf32 liblmdb0 libvirt0 \
    libcurl4 libfftw3-3 libopencv-core4.6 \
    qemu-system-x86 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install/usr/local /usr/local
COPY config/*.conf /etc/nikola/

RUN mkdir -p /var/lib/nikola/state /tmp/nikola

HEALTHCHECK CMD /usr/local/bin/twi-ctl status || exit 1

ENTRYPOINT ["/usr/local/bin/nikola-daemon"]
```

---

## G.2 Docker Compose Configuration

### G.2.1 Standard Deployment

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  nikola-spine:
    image: nikola:latest
    container_name: nikola-core
    build:
      context: .
      dockerfile: Dockerfile
      args:
        - DEBIAN_FRONTEND=noninteractive

    # Mount volumes for persistence
    volumes:
      - nikola-state:/var/lib/nikola/state
      - nikola-ingest:/var/lib/nikola/ingest
      - nikola-archive:/var/lib/nikola/archive
      - nikola-logs:/var/log/nikola
      - /tmp/nikola:/tmp/nikola  # IPC sockets

    # Environment variables (API keys)
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - NIKOLA_LOG_LEVEL=INFO

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '16.0'
          memory: 64G
        reservations:
          cpus: '8.0'
          memory: 32G

    # Restart policy
    restart: unless-stopped

    # Health check
    healthcheck:
      test: ["CMD", "/usr/local/bin/twi-ctl", "status"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s

    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"

    # Network
    networks:
      - nikola-net

# Named volumes
volumes:
  nikola-state:
    driver: local
  nikola-ingest:
    driver: local
  nikola-archive:
    driver: local
  nikola-logs:
    driver: local

# Network
networks:
  nikola-net:
    driver: bridge
```

### G.2.2 GPU-Accelerated Deployment

**File:** `docker-compose.cuda.yml`

```yaml
version: '3.8'

services:
  nikola-spine:
    image: nikola:cuda
    container_name: nikola-core-gpu
    build:
      context: .
      dockerfile: Dockerfile.cuda

    volumes:
      - nikola-state:/var/lib/nikola/state
      - nikola-ingest:/var/lib/nikola/ingest
      - /tmp/nikola:/tmp/nikola

    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - CUDA_VISIBLE_DEVICES=0  # Use GPU 0

    # GPU access
    deploy:
      resources:
        limits:
          memory: 64G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu, compute]

    restart: unless-stopped
    networks:
      - nikola-net

volumes:
  nikola-state:
  nikola-ingest:

networks:
  nikola-net:
```

---

## G.3 Build and Deployment Commands

### G.3.1 Initial Build

```bash
# Clone repository
git clone https://github.com/your-org/nikola.git
cd nikola

# Set API keys
export TAVILY_API_KEY="your-key-here"
export FIRECRAWL_API_KEY="your-key-here"
export GEMINI_API_KEY="your-key-here"

# Save to .env file for Docker Compose
cat > .env <<EOF
TAVILY_API_KEY=${TAVILY_API_KEY}
FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
GEMINI_API_KEY=${GEMINI_API_KEY}
EOF

# Build image
docker-compose build

# Expected output:
# [+] Building 1234.5s (23/23) FINISHED
# Successfully tagged nikola:latest
```

### G.3.2 Start Services

```bash
# Start in background
docker-compose up -d

# Check status
docker-compose ps

# Expected output:
# NAME              STATUS              PORTS
# nikola-core       Up 2 minutes        5555-5556/tcp

# View logs
docker-compose logs -f nikola-spine
```

### G.3.3 GPU Deployment

```bash
# Build CUDA image
docker-compose -f docker-compose.cuda.yml build

# Start with GPU
docker-compose -f docker-compose.cuda.yml up -d

# Verify GPU access
docker exec nikola-core-gpu nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 12.2     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA RTX 4090     Off  | 00000000:01:00.0 Off |                  Off |
# | 30%   45C    P0    70W / 450W |    512MiB / 24564MiB |      5%      Default |
# +-------------------------------+----------------------+----------------------+
```

### G.3.4 CLI Access

```bash
# Execute CLI commands inside container
docker exec nikola-core /usr/local/bin/twi-ctl status

# Or create alias for convenience
alias twi-ctl='docker exec nikola-core /usr/local/bin/twi-ctl'

# Now use normally
twi-ctl query "What is the golden ratio?"
twi-ctl nap
twi-ctl metrics
```

---

## G.4 Volume Management

### G.4.1 Backup State

```bash
# Create backup of persistent state
docker run --rm \
    -v nikola-state:/source \
    -v $(pwd)/backups:/backup \
    ubuntu:24.04 \
    tar czf /backup/nikola-state-$(date +%Y%m%d-%H%M%S).tar.gz -C /source .

# Backup to remote storage (AWS S3)
aws s3 cp backups/nikola-state-20241201-120000.tar.gz \
    s3://my-bucket/nikola/backups/
```

### G.4.2 Restore State

```bash
# Stop container
docker-compose down

# Restore from backup
docker run --rm \
    -v nikola-state:/target \
    -v $(pwd)/backups:/backup \
    ubuntu:24.04 \
    tar xzf /backup/nikola-state-20241201-120000.tar.gz -C /target

# Restart
docker-compose up -d
```

### G.4.3 Inspect Volume

```bash
# List files in volume
docker run --rm \
    -v nikola-state:/data \
    ubuntu:24.04 \
    ls -lh /data

# Expected output:
# -rw------- 1 root root 128M Dec  1 12:00 nikola_20241201_120000.nik
# -rw------- 1 root root  42M Dec  1 11:30 nikola_20241201_113000.nik
# -rw------- 1 root root  15K Dec  1 12:00 identity.json
```

---

## G.5 Resource Monitoring

### G.5.1 Container Stats

```bash
# Real-time resource usage
docker stats nikola-core

# Output:
# CONTAINER    CPU %    MEM USAGE / LIMIT    MEM %    NET I/O         BLOCK I/O
# nikola-core  12.5%    8.2GB / 64GB         12.8%    1.2MB / 850kB   45MB / 12MB
```

### G.5.2 Detailed Metrics

```bash
# Get JSON metrics
docker exec nikola-core twi-ctl metrics --json

# Parse with jq
docker exec nikola-core twi-ctl metrics --json | jq '.physics.avg_step_ms'

# Output: 0.48
```

### G.5.3 Health Checks

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' nikola-core

# Output: healthy

# View health check logs
docker inspect --format='{{json .State.Health}}' nikola-core | jq .
```

---

## G.6 Networking

### G.6.1 IPC Socket Access

**Host → Container:**

```bash
# Mount /tmp/nikola as volume
# CLI on host can communicate via IPC sockets

# On host:
./twi-ctl-host status

# Connects to: /tmp/nikola/spine_frontend.ipc (mounted from container)
```

### G.6.2 TCP Socket Configuration

**For remote access, use TCP instead of IPC:**

```yaml
# docker-compose.yml
services:
  nikola-spine:
    ports:
      - "5555:5555"  # Frontend
      - "5556:5556"  # Backend
    environment:
      - NIKOLA_TRANSPORT=tcp
      - NIKOLA_BIND_ADDRESS=0.0.0.0
```

**Client Configuration:**

```cpp
// Change from IPC to TCP
socket.connect("tcp://nikola-server:5555");
```

---

## G.7 Production Best Practices

### G.7.1 Multi-Container Architecture

**Separate services for scalability:**

```yaml
services:
  # Spine broker (message router)
  nikola-spine:
    image: nikola:spine
    ports:
      - "5555:5555"

  # Physics engine (stateless, can scale horizontally)
  nikola-physics:
    image: nikola:physics
    deploy:
      replicas: 4
    depends_on:
      - nikola-spine

  # Memory system (persistent state)
  nikola-memory:
    image: nikola:memory
    volumes:
      - nikola-state:/var/lib/nikola/state
    depends_on:
      - nikola-spine

  # Orchestrator (coordinator)
  nikola-orchestrator:
    image: nikola:orchestrator
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
    depends_on:
      - nikola-spine
      - nikola-physics
      - nikola-memory
```

### G.7.2 Secrets Management

**Use Docker secrets (Swarm mode):**

```yaml
services:
  nikola-spine:
    secrets:
      - tavily_key
      - firecrawl_key
      - gemini_key
    environment:
      - TAVILY_API_KEY_FILE=/run/secrets/tavily_key

secrets:
  tavily_key:
    external: true
  firecrawl_key:
    external: true
  gemini_key:
    external: true
```

**Create secrets:**

```bash
echo "your-tavily-key" | docker secret create tavily_key -
echo "your-firecrawl-key" | docker secret create firecrawl_key -
echo "your-gemini-key" | docker secret create gemini_key -
```

### G.7.3 Logging and Monitoring

**Centralized logging with ELK stack:**

```yaml
services:
  nikola-spine:
    logging:
      driver: "gelf"
      options:
        gelf-address: "udp://logstash:12201"
        tag: "nikola"

  logstash:
    image: docker.elastic.co/logstash/logstash:8.10.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "12201:12201/udp"

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.10.0
    environment:
      - discovery.type=single-node

  kibana:
    image: docker.elastic.co/kibana/kibana:8.10.0
    ports:
      - "5601:5601"
```

---

**Cross-References:**
- See Section 9.4 for build system details
- See Appendix E for troubleshooting Docker issues
- See Appendix F for security hardening
- See official Docker documentation: https://docs.docker.com/


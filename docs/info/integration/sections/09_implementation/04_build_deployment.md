# BUILD AND DEPLOYMENT

## 25.1 CLI Controller

**Binary Name:** `twi-ctl` (Toroidal Waveform Intelligence Controller)

**Usage:**

```bash
twi-ctl <command> [arguments]
```

### Command Set

| Command | Arguments | Description |
|---------|-----------|-------------|
| `query` | `"<text>"` | Submit query to system |
| `status` | - | Show system status (dopamine, boredom, active nodes) |
| `nap` | - | Trigger immediate nap/checkpoint |
| `train` | `[mamba\|transformer\|both]` | Trigger training session |
| `ingest` | `<file_path>` | Manually ingest file |
| `export` | `<output.gguf>` | Export to GGUF format |
| `goals` | `list\|add\|complete` | Manage goal system |
| `identity` | - | Show identity profile |
| `firewall` | `add <pattern>` | Add hazardous pattern |
| `metrics` | - | Show performance metrics |
| `shutdown` | - | Graceful shutdown |

### Implementation Excerpt

```cpp
// File: tools/twi-ctl/main.cpp

class TWIController {
    zmq::context_t ctx;
    zmq::socket_t socket;

public:
    TWIController() : ctx(1), socket(ctx, ZMQ_REQ) {
        socket.connect("ipc:///tmp/nikola/spine_cli.ipc");
    }

    std::string send_query(const std::string& query_text) {
        NeuralSpike spike;
        spike.set_request_id(generate_uuid());
        spike.set_timestamp(current_timestamp());
        spike.set_sender(ComponentID::CLI_CONTROLLER);
        spike.set_recipient(ComponentID::ORCHESTRATOR);
        spike.set_text_data(query_text);

        // Serialize directly to ZMQ message (zero-copy, no intermediate std::string)
        size_t msg_size = spike.ByteSizeLong();
        zmq::message_t request(msg_size);
        spike.SerializeToArray(request.data(), msg_size);
        socket.send(request, zmq::send_flags::none);

        // Receive response
        zmq::message_t reply;
        socket.recv(reply, zmq::recv_flags::none);

        NeuralSpike response;
        response.ParseFromArray(reply.data(), reply.size());

        return response.text_data();
    }
};

// Main entry point with proper libcurl initialization
int main(int argc, char* argv[]) {
    // CRITICAL: Initialize libcurl globally before any threading or network operations
    // This prevents race conditions with the CustomHTTPClient used by external tools
    // See Section 12.4 for CustomHTTPClient implementation
    curl_global_init(CURL_GLOBAL_ALL);

    // Ensure cleanup on exit
    std::atexit([]() {
        curl_global_cleanup();
    });

    // Parse command and execute
    if (argc < 2) {
        std::cerr << "Usage: twi-ctl <command> [args...]" << std::endl;
        return 1;
    }

    TWIController controller;
    std::string command = argv[1];

    if (command == "query" && argc == 3) {
        std::string result = controller.send_query(argv[2]);
        std::cout << result << std::endl;
    } else if (command == "status") {
        // ... other commands ...
    } else {
        std::cerr << "Unknown command: " << command << std::endl;
        return 1;
    }

    // libcurl will be cleaned up automatically via std::atexit
    return 0;
}
```

## 25.2 Build System (CMake)

### Root CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(Nikola VERSION 0.0.4 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build types
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Find dependencies
find_package(ZeroMQ REQUIRED)
find_package(Protobuf REQUIRED)
find_package(LMDB REQUIRED)
find_package(libvirt REQUIRED)
find_package(FFTW3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.11.0 REQUIRED)  # JSON library for configuration
find_package(CUDA QUIET)

# Optional AVX-512
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    add_compile_options(-mavx512f)
    add_definitions(-DUSE_AVX512)
endif()

# Subdirectories
add_subdirectory(proto)
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(tests)
```

### Library CMakeLists.txt

```cmake
# src/CMakeLists.txt

add_library(lib9dtwi SHARED
    types/nit.cpp
    types/coord9d.cpp
    physics/torus_manifold.cpp
    physics/emitter_array.cpp
    physics/wave_engine.cpp
    physics/shvo_grid.cpp
    mamba/hilbert_scan.cpp
    mamba/ssm_kernel.cpp
    reasoning/transformer.cpp
    reasoning/wave_attention.cpp
    reasoning/embedder.cpp
    spine/broker.cpp
    spine/component_client.cpp
    spine/shadow_spine.cpp
    orchestrator/smart_router.cpp
    agents/tavily.cpp
    agents/firecrawl.cpp
    agents/gemini.cpp
    agents/http_client.cpp
    executor/kvm_executor.cpp
    autonomy/engs.cpp
    autonomy/dopamine.cpp
    autonomy/boredom.cpp
    autonomy/goals.cpp
    autonomy/trainers.cpp
    autonomy/dream_weave.cpp
    persistence/lsm_dmc.cpp
    persistence/gguf_export.cpp
    persistence/identity.cpp
    multimodal/audio_resonance.cpp
    multimodal/visual_cymatics.cpp
    security/resonance_firewall.cpp
    security/csvp.cpp
    self_improve/profiler.cpp
    self_improve/adversarial_dojo.cpp
    ingestion/sentinel.cpp
)

target_link_libraries(lib9dtwi
    PUBLIC
        zmq
        protobuf
        lmdb
        virt
        fftw3
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json  # JSON library for configuration
)

target_include_directories(lib9dtwi
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

# CUDA kernels (if available)
if(CUDA_FOUND)
    cuda_add_library(nikola_cuda STATIC
        physics/kernels/wave_propagate.cu
    )
    target_link_libraries(lib9dtwi PUBLIC nikola_cuda)
endif()
```

## 25.3 Docker Deployment

### Multi-Stage Dockerfile

```dockerfile
# Stage 1: Build environment
FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libzmq3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    liblmdb-dev \
    libvirt-dev \
    libfftw3-dev \
    libopencv-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency manifests first (for cache optimization)
COPY CMakeLists.txt .
COPY proto/ proto/

# Configure CMake dependencies layer (cached unless CMakeLists.txt changes)
RUN cmake -DCMAKE_BUILD_TYPE=Release -B build

# Copy source code (invalidates cache only when source changes)
COPY src/ src/
COPY include/ include/

# Build application (cached unless source or dependencies change)
RUN cmake --build build --parallel $(nproc) && \
    cmake --install build --prefix /install

# Stage 2: Runtime environment
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    libzmq5 \
    libprotobuf32 \
    liblmdb0 \
    libvirt0 \
    libfftw3-3 \
    libopencv-core4.6 \
    libcurl4 \
    qemu-system-x86 \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify runtime dependencies with ldd during build:
# RUN ldd /usr/local/bin/nikola-daemon && ldd /usr/local/bin/twi-ctl

COPY --from=builder /install /usr/local

# Create directories
RUN mkdir -p /var/lib/nikola/{state,ingest,archive} && \
    mkdir -p /etc/nikola

# Copy config
COPY config/*.conf /etc/nikola/

# Expose IPC socket
VOLUME ["/tmp/nikola"]

ENTRYPOINT ["/usr/local/bin/nikola-daemon"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  nikola-spine:
    image: nikola:latest
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - nikola-state:/var/lib/nikola/state
      - nikola-ingest:/var/lib/nikola/ingest
      - /tmp/nikola:/tmp/nikola
    environment:
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - FIRECRAWL_API_KEY=${FIRECRAWL_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  nikola-state:
  nikola-ingest:
```

## 25.4 Running the System

### Start Services

```bash
# Start Docker compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f nikola-spine
```

### CLI Usage Examples

```bash
# Query the system
twi-ctl query "What is the golden ratio?"

# Check system status
twi-ctl status

# Trigger nap
twi-ctl nap

# Start training
twi-ctl train both

# Manually ingest a file
twi-ctl ingest /path/to/document.pdf

# Export to GGUF
twi-ctl export nikola-snapshot.gguf

# Manage goals
twi-ctl goals list
twi-ctl goals add "Learn quantum computing"
twi-ctl goals complete <goal-id>

# View identity
twi-ctl identity

# Add firewall pattern
twi-ctl firewall add "ignore previous instructions"

# View metrics
twi-ctl metrics

# Shutdown
twi-ctl shutdown
```

## 25.5 Testing

### Unit Tests

```bash
# Run all unit tests
cd build
ctest --output-on-failure

# Run specific test suite
ctest -R test_nonary

# Run with Valgrind (memory check)
ctest -T memcheck
```

### Integration Tests

```bash
# Run integration tests
ctest -R integration

# Benchmark performance
ctest -R bench
```

### Physics Invariants Check

```bash
# Verify energy conservation
./build/tests/unit/test_energy_conservation

# Verify nonary arithmetic
./build/tests/unit/test_nonary

# Verify toroidal wrapping
./build/tests/unit/test_coord9d
```

## 25.6 Deployment Checklist

**Pre-Deployment:**
- [ ] All unit tests pass (100%)
- [ ] All integration tests pass
- [ ] Physics invariants verified
- [ ] Security verification passed (Appendix G)
- [ ] Performance benchmarks met (Appendix F)
- [ ] Docker image builds successfully

**Deployment:**
- [ ] Configure API keys in environment
- [ ] Set up persistence volumes
- [ ] Configure firewall rules
- [ ] Start services with docker-compose
- [ ] Verify CLI connectivity

**Post-Deployment:**
- [ ] Monitor system status
- [ ] Check logs for errors
- [ ] Verify external tool connectivity
- [ ] Test basic query/response
- [ ] Verify nap/checkpoint cycle

## 25.7 Monitoring

### System Metrics

```bash
# Dopamine level
twi-ctl status | grep Dopamine

# Active nodes count
twi-ctl status | grep "Active Nodes"

# Uptime
twi-ctl status | grep Uptime
```

### Performance Metrics

```bash
# Detailed metrics
twi-ctl metrics

# Output includes:
# - Wave propagation time
# - Resonance detection latency
# - Training cycle duration
# - Memory usage
# - GPU utilization (if available)
```

---

**Cross-References:**
- See Section 10 for ZeroMQ Spine details
- See Section 26 for File Structure
- See Section 28 for Implementation Checklist
- See Appendix I for Docker deployment details

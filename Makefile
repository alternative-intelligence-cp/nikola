# ==============================================================================
# Nikola — Makefile
# Docker build, test, and deployment targets
# ==============================================================================

NIKOLA_VERSION ?= latest
REGISTRY       ?= ghcr.io/alternative-intelligence-cp
IMAGE          := $(REGISTRY)/nikola
CUDA_ARCH      ?= 86
COMPOSE        := docker compose

# ─── Docker Build ─────────────────────────────────────────────────────────────

.PHONY: docker-build docker-build-cpu docker-push docker-run docker-stop
.PHONY: compose-up compose-down compose-gpu compose-logs compose-ps
.PHONY: native-build native-test clean help

## Build GPU-enabled Docker image (default)
docker-build:
	docker build \
		--build-arg CUDA_ARCH=$(CUDA_ARCH) \
		-t $(IMAGE):$(NIKOLA_VERSION) \
		-t $(IMAGE):latest \
		.

## Build CPU-only Docker image
docker-build-cpu:
	docker build \
		--build-arg BUILDER_IMAGE=ubuntu:24.04 \
		--build-arg RUNTIME_IMAGE=ubuntu:24.04 \
		--build-arg CUDA_ARCH="" \
		-t $(IMAGE):$(NIKOLA_VERSION)-cpu \
		.

## Push image to registry
docker-push:
	docker push $(IMAGE):$(NIKOLA_VERSION)
	docker push $(IMAGE):latest

## Run standalone spine container with GPU
docker-run:
	docker run --gpus all -it --rm \
		-p 5555:5555 -p 5556:5556 -p 5557:5557 \
		-p 5560:5560 -p 5561:5561 -p 9876:9876 \
		-v nikola-data:/data/nikola \
		$(IMAGE):$(NIKOLA_VERSION)

## Stop all nikola containers
docker-stop:
	$(COMPOSE) down

# ─── Docker Compose ───────────────────────────────────────────────────────────

## Start all services (CPU)
compose-up:
	NIKOLA_VERSION=$(NIKOLA_VERSION) $(COMPOSE) up -d

## Start all services with GPU support
compose-gpu:
	NIKOLA_VERSION=$(NIKOLA_VERSION) $(COMPOSE) --profile gpu up -d

## Stop all services (graceful shutdown)
compose-down:
	$(COMPOSE) down

## View service logs
compose-logs:
	$(COMPOSE) logs -f

## Show service status
compose-ps:
	$(COMPOSE) ps

## Run diagnostics container
compose-diag:
	NIKOLA_VERSION=$(NIKOLA_VERSION) $(COMPOSE) --profile diagnostics run --rm nikola-diag

# ─── Native Build ─────────────────────────────────────────────────────────────

## Build natively (host toolchain)
native-build:
	@mkdir -p build && cd build && cmake .. && make -j$$(nproc)

## Run native test suite
native-test:
	@cd build && ctest --output-on-failure -j$$(nproc)

## Clean build artifacts
clean:
	rm -rf build/

# ─── Help ─────────────────────────────────────────────────────────────────────

## Show available targets
help:
	@echo "Nikola Build Targets:"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build      Build GPU Docker image"
	@echo "    docker-build-cpu  Build CPU-only Docker image"
	@echo "    docker-push       Push to container registry"
	@echo "    docker-run        Run standalone spine (GPU)"
	@echo "    docker-stop       Stop all containers"
	@echo ""
	@echo "  Compose:"
	@echo "    compose-up        Start all services (CPU)"
	@echo "    compose-gpu       Start with GPU support"
	@echo "    compose-down      Graceful shutdown"
	@echo "    compose-logs      Follow service logs"
	@echo "    compose-ps        Show service status"
	@echo "    compose-diag      Run diagnostics"
	@echo ""
	@echo "  Native:"
	@echo "    native-build      Build with host toolchain"
	@echo "    native-test       Run test suite"
	@echo "    clean             Remove build artifacts"

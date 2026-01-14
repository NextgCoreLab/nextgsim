# nextgsim Makefile
# Pure Rust 5G UE and gNB Simulator

.PHONY: all build release debug test clean fmt lint check doc \
        docker docker-gnb docker-ue docker-push docker-clean \
        run-gnb run-ue install help

# Configuration
CARGO := cargo
DOCKER := docker
DOCKER_COMPOSE := docker compose
IMAGE_PREFIX := nextgsim
VERSION := $(shell grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
REGISTRY ?= ghcr.io/nextgsim

# Default target
all: build

# =============================================================================
# Build Targets
# =============================================================================

## Build all binaries (debug mode)
build:
	$(CARGO) build --workspace

## Build all binaries (release mode)
release:
	$(CARGO) build --release --workspace

## Build debug binaries
debug:
	$(CARGO) build --workspace

## Build only gNB binary
build-gnb:
	$(CARGO) build --release --package nextgsim-gnb

## Build only UE binary
build-ue:
	$(CARGO) build --release --package nextgsim-ue

## Build only CLI binary
build-cli:
	$(CARGO) build --release --package nextgsim-cli

# =============================================================================
# Test Targets
# =============================================================================

## Run all tests
test:
	$(CARGO) test --workspace

## Run tests with output
test-verbose:
	$(CARGO) test --workspace -- --nocapture

## Run unit tests only (exclude integration tests)
test-unit:
	$(CARGO) test --workspace --lib

## Run integration tests only
test-integration:
	$(CARGO) test --package tests

## Run tests for a specific crate
test-%:
	$(CARGO) test --package nextgsim-$*

## Run crypto test vectors
test-crypto:
	$(CARGO) test --package nextgsim-crypto -- --nocapture

# =============================================================================
# Code Quality
# =============================================================================

## Format code
fmt:
	$(CARGO) fmt --all

## Check formatting
fmt-check:
	$(CARGO) fmt --all -- --check

## Run clippy lints
lint:
	$(CARGO) clippy --workspace --all-targets -- -D warnings

## Run all checks (fmt + lint + test)
check: fmt-check lint test

## Generate documentation
doc:
	$(CARGO) doc --workspace --no-deps

## Open documentation in browser
doc-open:
	$(CARGO) doc --workspace --no-deps --open

# =============================================================================
# Docker Targets
# =============================================================================

## Build all Docker images
docker: docker-gnb docker-ue docker-all

## Build combined Docker image (all binaries)
docker-all:
	$(DOCKER) build -t $(IMAGE_PREFIX):$(VERSION) -t $(IMAGE_PREFIX):latest .

## Build gNB Docker image
docker-gnb:
	$(DOCKER) build -f Dockerfile.gnb -t $(IMAGE_PREFIX)-gnb:$(VERSION) -t $(IMAGE_PREFIX)-gnb:latest .

## Build UE Docker image
docker-ue:
	$(DOCKER) build -f Dockerfile.ue -t $(IMAGE_PREFIX)-ue:$(VERSION) -t $(IMAGE_PREFIX)-ue:latest .

## Push Docker images to registry
docker-push:
	$(DOCKER) tag $(IMAGE_PREFIX):$(VERSION) $(REGISTRY)/$(IMAGE_PREFIX):$(VERSION)
	$(DOCKER) tag $(IMAGE_PREFIX)-gnb:$(VERSION) $(REGISTRY)/$(IMAGE_PREFIX)-gnb:$(VERSION)
	$(DOCKER) tag $(IMAGE_PREFIX)-ue:$(VERSION) $(REGISTRY)/$(IMAGE_PREFIX)-ue:$(VERSION)
	$(DOCKER) push $(REGISTRY)/$(IMAGE_PREFIX):$(VERSION)
	$(DOCKER) push $(REGISTRY)/$(IMAGE_PREFIX)-gnb:$(VERSION)
	$(DOCKER) push $(REGISTRY)/$(IMAGE_PREFIX)-ue:$(VERSION)

## Clean Docker images
docker-clean:
	$(DOCKER) rmi -f $(IMAGE_PREFIX):$(VERSION) $(IMAGE_PREFIX):latest 2>/dev/null || true
	$(DOCKER) rmi -f $(IMAGE_PREFIX)-gnb:$(VERSION) $(IMAGE_PREFIX)-gnb:latest 2>/dev/null || true
	$(DOCKER) rmi -f $(IMAGE_PREFIX)-ue:$(VERSION) $(IMAGE_PREFIX)-ue:latest 2>/dev/null || true

# =============================================================================
# Docker Compose Targets
# =============================================================================

## Start all services with docker compose
compose-up:
	$(DOCKER_COMPOSE) up -d

## Stop all services
compose-down:
	$(DOCKER_COMPOSE) down

## View logs
compose-logs:
	$(DOCKER_COMPOSE) logs -f

## Rebuild and restart services
compose-rebuild:
	$(DOCKER_COMPOSE) up -d --build

# =============================================================================
# Run Targets
# =============================================================================

## Run gNB with default config
run-gnb:
	$(CARGO) run --release --package nextgsim-gnb -- -c ../config/open5gs-gnb.yaml

## Run UE with default config
run-ue:
	$(CARGO) run --release --package nextgsim-ue -- -c ../config/open5gs-ue.yaml

## Run CLI
run-cli:
	$(CARGO) run --release --package nextgsim-cli

# =============================================================================
# Install Targets
# =============================================================================

## Install binaries to ~/.cargo/bin
install:
	$(CARGO) install --path nextgsim-gnb
	$(CARGO) install --path nextgsim-ue
	$(CARGO) install --path nextgsim-cli

## Install to /usr/local/bin (requires sudo)
install-system: release
	sudo install -m 755 target/release/nextgsim-gnb /usr/local/bin/
	sudo install -m 755 target/release/nextgsim-ue /usr/local/bin/
	sudo install -m 755 target/release/nextgsim-cli /usr/local/bin/

# =============================================================================
# Clean Targets
# =============================================================================

## Clean build artifacts
clean:
	$(CARGO) clean

## Clean everything including Docker
clean-all: clean docker-clean

# =============================================================================
# Development Helpers
# =============================================================================

## Watch for changes and rebuild
watch:
	$(CARGO) watch -x build

## Watch for changes and run tests
watch-test:
	$(CARGO) watch -x test

## Update dependencies
update:
	$(CARGO) update

## Check for outdated dependencies
outdated:
	$(CARGO) outdated

## Security audit
audit:
	$(CARGO) audit

# =============================================================================
# Help
# =============================================================================

## Show this help
help:
	@echo "nextgsim - Pure Rust 5G UE and gNB Simulator"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Build Targets:"
	@echo "  build          Build all binaries (debug mode)"
	@echo "  release        Build all binaries (release mode)"
	@echo "  build-gnb      Build only gNB binary"
	@echo "  build-ue       Build only UE binary"
	@echo "  build-cli      Build only CLI binary"
	@echo ""
	@echo "Test Targets:"
	@echo "  test           Run all tests"
	@echo "  test-verbose   Run tests with output"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests"
	@echo "  test-crypto    Run crypto test vectors"
	@echo ""
	@echo "Code Quality:"
	@echo "  fmt            Format code"
	@echo "  lint           Run clippy lints"
	@echo "  check          Run all checks (fmt + lint + test)"
	@echo "  doc            Generate documentation"
	@echo ""
	@echo "Docker:"
	@echo "  docker         Build all Docker images"
	@echo "  docker-gnb     Build gNB Docker image"
	@echo "  docker-ue      Build UE Docker image"
	@echo "  docker-push    Push images to registry"
	@echo "  compose-up     Start services with docker compose"
	@echo "  compose-down   Stop services"
	@echo ""
	@echo "Run:"
	@echo "  run-gnb        Run gNB with default config"
	@echo "  run-ue         Run UE with default config"
	@echo "  run-cli        Run CLI tool"
	@echo ""
	@echo "Install:"
	@echo "  install        Install to ~/.cargo/bin"
	@echo "  install-system Install to /usr/local/bin"
	@echo ""
	@echo "Clean:"
	@echo "  clean          Clean build artifacts"
	@echo "  clean-all      Clean everything including Docker"

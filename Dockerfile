# nextgsim - Pure Rust 5G UE and gNB Simulator
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM rust:1.83-bookworm AS builder

# Install rustfmt for ASN.1 code generation
RUN rustup component add rustfmt

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY rustfmt.toml ./

# Copy all crate directories
COPY nextgsim-common ./nextgsim-common
COPY nextgsim-crypto ./nextgsim-crypto
COPY nextgsim-nas ./nextgsim-nas
COPY nextgsim-ngap ./nextgsim-ngap
COPY nextgsim-rrc ./nextgsim-rrc
COPY nextgsim-rls ./nextgsim-rls
COPY nextgsim-gtp ./nextgsim-gtp
COPY nextgsim-sctp ./nextgsim-sctp
COPY nextgsim-gnb ./nextgsim-gnb
COPY nextgsim-ue ./nextgsim-ue
COPY nextgsim-cli ./nextgsim-cli
COPY tests ./tests
COPY tools ./tools

# Build release binaries
RUN cargo build --release

# =============================================================================
# Stage 2: Runtime (minimal image)
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    iproute2 \
    iptables \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash nextgsim

# Create directories
RUN mkdir -p /etc/nextgsim /var/log/nextgsim \
    && chown -R nextgsim:nextgsim /etc/nextgsim /var/log/nextgsim

# Copy binaries from builder
COPY --from=builder /app/target/release/nr-gnb /usr/local/bin/
COPY --from=builder /app/target/release/nr-ue /usr/local/bin/
COPY --from=builder /app/target/release/nr-cli /usr/local/bin/

# Set capabilities for TUN interface (requires privileged or CAP_NET_ADMIN)
# Note: This is set at runtime, not build time

WORKDIR /etc/nextgsim

# Default to running as root for TUN interface access
# Use --user flag to run as nextgsim when TUN is not needed
USER root

# Expose CLI ports (default range)
EXPOSE 4997-5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nr-cli --dump || exit 1

# Default command (override with docker run)
CMD ["nr-gnb", "-c", "/etc/nextgsim/gnb.yaml"]

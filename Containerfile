# Cross-compile for the image target without running the target image.
FROM --platform=$BUILDPLATFORM rust:1-bookworm AS build

ARG TARGETARCH
WORKDIR /src

RUN apt-get update \
    && if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get install -y --no-install-recommends ca-certificates gcc-aarch64-linux-gnu libc6-dev-arm64-cross; \
        rustup target add aarch64-unknown-linux-gnu; \
    else \
        apt-get install -y --no-install-recommends ca-certificates; \
        rustup target add x86_64-unknown-linux-gnu; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN set -eux; \
    if [ "$TARGETARCH" = "arm64" ]; then \
        export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc; \
        export AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar; \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc; \
        target=aarch64-unknown-linux-gnu; \
    else \
        target=x86_64-unknown-linux-gnu; \
    fi; \
    cargo build --release --locked --target "$target" --bin evelyn-serve; \
    cp "target/${target}/release/evelyn-serve" /tmp/evelyn-serve

FROM debian:bookworm-slim AS runtime

ARG TARGETARCH
LABEL org.opencontainers.image.source="https://github.com/neuralmimicry/evelyn"

COPY --from=build /tmp/evelyn-serve /usr/local/bin/evelyn-serve
COPY --from=build /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt

USER 65532:65532
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD ["/usr/local/bin/evelyn-serve", "--healthcheck"]
ENTRYPOINT ["/usr/local/bin/evelyn-serve"]

#!/usr/bin/env bash
set -x
set -e

SYSROOT=/valida-toolchain

if [ "$(uname -m)" = "x86_64" ]; then
  BUILD_COMMAND=(cargo multiarch)
  TARGET_BINARY="./target/cargo-multiarch/x86_64-unknown-linux-gnu/release/valida"
else
  BUILD_COMMAND=(cargo build --release)
  TARGET_BINARY="./target/release/valida"
fi

# Check if DEBUG_ASSERTIONS_ENABLED is set to true
if [ "$DEBUG_ASSERTIONS_ENABLED" = true ]; then
    RUSTFLAGS="-C debug-assertions" "${BUILD_COMMAND[@]}"
else
    "${BUILD_COMMAND[@]}"
fi

mkdir -p "$SYSROOT/bin"
install "$TARGET_BINARY" "$SYSROOT/bin/valida"

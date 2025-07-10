This package exposes WASM module for Valida runner, prover and verifier.

# Installation of dependencies

1. Install `wasm32` support: `rustup target add wasm32-unknown-unknown`
2. [Install](https://rustwasm.github.io/wasm-pack/installer/) `wasm-pack`
  - `cargo install wasm-pack`
  - or `curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sudo sh`
3. [Install](https://nodejs.org/en/download) `node.js` version v22.13.1 (LTS).
  - Download `node.js`
    - on `x86`: `wget https://nodejs.org/dist/v22.13.1/node-v22.13.1-linux-x64.tar.xz`
    - on `arm`: `wget https://nodejs.org/dist/v22.13.1/node-v22.13.1-linux-arm64.tar.xz`
  - then untar and add its `bin` directory to `PATH`.
  - Alternatively, install `node.js` v22.13.1 [using nvm](https://github.com/nvm-sh/nvm?tab=readme-ov-file#installing-and-updating).

`wasm-pack` is required to build WASM bindings.
`node.js` is required to run the tests.

# Building

In the package main directory:

```
wasm-pack build --release
```

# Testing

In the package main directory:

```
wasm-pack test --release --node
```

Tests are built in the release mode to save CI time.
Tests are run in node.js.

# Releasing

To release this package on `npm`, follow these steps:

1. Increment the package version in the `Cargo.toml` file to match the main Valida release version.
2. Build the package as specified above in the Building section
3. `wasm-pack login`
4. `cd pkg`
5. `npm publish --access=public`

# Resources

For more information on WASM, Rust to WASM compilation and testing see:
- https://rustwasm.github.io/docs/wasm-bindgen/
- https://rustwasm.github.io/docs/book/game-of-life/hello-world.html
- https://rustwasm.github.io/docs/wasm-pack/
- https://doc.rust-lang.org/rustc/platform-support/wasm32-unknown-unknown.html
- https://rustwasm.github.io/wasm-pack/book/tutorials/npm-browser-packages/building-your-project.html

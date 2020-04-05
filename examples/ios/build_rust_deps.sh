#!/bin/sh

set -e

PATH=$PATH:$HOME/.cargo/bin
echo "$PATH"
cargo build --example ocean_ios --features metal --target aarch64-apple-ios
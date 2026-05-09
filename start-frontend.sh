#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/frontend"
pnpm dev --host 0.0.0.0

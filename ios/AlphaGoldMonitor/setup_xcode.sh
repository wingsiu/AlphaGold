#!/bin/bash
# Generate AlphaGoldMonitor.xcodeproj from project.yml (requires xcodegen).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "Installing xcodegen via Homebrew…"
  brew install xcodegen
fi

xcodegen generate
echo "Open: $DIR/AlphaGoldMonitor.xcodeproj"
echo "Set DEVELOPMENT_TEAM in Xcode → Signing & Capabilities."

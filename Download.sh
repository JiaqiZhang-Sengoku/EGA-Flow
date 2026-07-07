#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$PROJECT_ROOT/Model_Checkpoints"
mkdir -p "$CHECKPOINT_DIR"

download_file() {
  url="$1"
  output="$2"
  if [ -z "$url" ]; then
    echo "Missing download URL for $output"
    exit 1
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$output"
  else
    echo "Please install wget or curl."
    exit 1
  fi
}

case "${1:-all}" in
  celeba)
    download_file "${CELEBA_URL:-}" "$CHECKPOINT_DIR/CelebA.pt"
    ;;
  afhq_cat)
    download_file "${AFHQ_CAT_URL:-}" "$CHECKPOINT_DIR/AFHQ-Cat.pt"
    ;;
  all)
    download_file "${CELEBA_URL:-}" "$CHECKPOINT_DIR/CelebA.pt"
    download_file "${AFHQ_CAT_URL:-}" "$CHECKPOINT_DIR/AFHQ-Cat.pt"
    ;;
  *)
    echo "Usage: CELEBA_URL=<url> AFHQ_CAT_URL=<url> bash Download.sh [all|celeba|afhq_cat]"
    exit 1
    ;;
esac

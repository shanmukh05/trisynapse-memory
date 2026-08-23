#!/bin/sh
set -eu

VERSION="${TRISYNAPSE_MEMORY_VERSION:-0.1.2}"
PACKAGE="trisynapse-memory[all]==$VERSION"
METADATA_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/trisynapse-memory"

say() { printf '%s\n' "$*"; }
fail() { say "trisynapse-memory installer: $*" >&2; exit 1; }

DETECTED_OS="$(uname -s)"
DETECTED_ARCH="$(uname -m)"
case "$DETECTED_OS" in
  Darwin|Linux) ;;
  *) fail "macOS and Linux are supported by install.sh; use install.ps1 on Windows" ;;
esac
case "$DETECTED_ARCH" in
  x86_64|amd64|arm64|aarch64) say "Detected ${DETECTED_OS} / ${DETECTED_ARCH}." ;;
  *) fail "unsupported CPU architecture: ${DETECTED_ARCH}" ;;
esac

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  say "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV="${UV_INSTALL_DIR:-$HOME/.local/bin}/uv"
  [ -x "$UV" ] || UV="$HOME/.cargo/bin/uv"
  [ -x "$UV" ] || fail "uv installed but was not found; add ~/.local/bin to PATH and rerun"
fi

say "Installing or upgrading ${PACKAGE}..."
INSTALL_ATTEMPT=1
until "$UV" tool install --upgrade "$PACKAGE"; do
  if [ "$INSTALL_ATTEMPT" -ge 5 ]; then
    fail "package did not become available after ${INSTALL_ATTEMPT} attempts: ${PACKAGE}"
  fi
  say "Package is not available yet; retrying in 6 seconds..."
  INSTALL_ATTEMPT=$((INSTALL_ATTEMPT + 1))
  sleep 6
done
mkdir -p "$METADATA_DIR"
chmod 700 "$METADATA_DIR"
{
  printf 'installed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'installer=github-release\n'
  printf 'version=%s\n' "$VERSION"
  printf 'os=%s\n' "$DETECTED_OS"
  printf 'architecture=%s\n' "$DETECTED_ARCH"
  printf 'uv=%s\n' "$UV"
} > "$METADATA_DIR/install.env"
chmod 600 "$METADATA_DIR/install.env"

UV_BIN_DIR="$("$UV" tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")"
case ":$PATH:" in
  *":${UV_BIN_DIR}:"*) ;;
  *) say "Add ${UV_BIN_DIR} to PATH, then open a new terminal." ;;
esac

if command -v trisynapse-memory >/dev/null 2>&1; then
  trisynapse-memory --json check
elif [ -x "$UV_BIN_DIR/trisynapse-memory" ]; then
  "$UV_BIN_DIR/trisynapse-memory" --json check
else
  fail "installation completed but trisynapse-memory was not found"
fi

say "Installed. Run: trisynapse-memory"

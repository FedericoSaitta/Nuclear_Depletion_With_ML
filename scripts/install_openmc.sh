#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build OpenMC's C++ core and install its Python API into the *simulation*
# uv environment (.venv-sim).
#
# OpenMC is not on PyPI, so it cannot live in uv.lock. It is
# treated as a system dependency, like CUDA: pinned by git tag here, installed
# with --no-deps so that every one of its Python requirements still comes from
# the locked `sim` extra.
#
# Prerequisites:
#   Debian/Ubuntu/WSL2 : sudo apt install -y g++ cmake libhdf5-dev git
#   Cluster            : module load gcc cmake hdf5      (site-specific)
#
# Usage:
#   ./scripts/install_openmc.sh                 # default tag
#   OPENMC_VERSION=v0.15.2 ./scripts/install_openmc.sh
# ---------------------------------------------------------------------------
set -euo pipefail

OPENMC_VERSION="${OPENMC_VERSION:-v0.15.2}"   # <-- pin. Record this in the README.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${OPENMC_SRC:-$REPO_ROOT/external/openmc}"
VENV="${OPENMC_TARGET_VENV:-$REPO_ROOT/.venv-sim}"
JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4)"

if [ ! -d "$VENV" ]; then
  echo "error: $VENV does not exist. Run this first:" >&2
  echo "  UV_PROJECT_ENVIRONMENT=.venv-sim uv sync --extra sim" >&2
  exit 1
fi

# --- 1. source, pinned -----------------------------------------------------
if [ ! -d "$SRC/.git" ]; then
  echo ">>> cloning OpenMC $OPENMC_VERSION into $SRC"
  git clone --recurse-submodules https://github.com/openmc-dev/openmc.git "$SRC"
fi
git -C "$SRC" fetch --tags --quiet
git -C "$SRC" checkout --quiet "$OPENMC_VERSION"
git -C "$SRC" submodule update --init --recursive --quiet

# --- 2. C++ core -----------------------------------------------------------
# The POST_BUILD step in OpenMC's CMakeLists copies libopenmc into
# $SRC/openmc/lib/, where its pyproject.toml picks it up as package data.
# That is what makes step 3 produce a self-contained, relocatable install.
echo ">>> configuring"
cmake -S "$SRC" -B "$SRC/build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$SRC/install" \
      -DOPENMC_USE_OPENMP=ON \
      -DOPENMC_USE_MPI=OFF

echo ">>> building with $JOBS jobs (expect 5-15 min)"
cmake --build "$SRC/build" -j"$JOBS"
cmake --install "$SRC/build"          # optional: gives you the `openmc` CLI

# sanity check before we install the Python side
if ! ls "$SRC"/openmc/lib/libopenmc.* >/dev/null 2>&1; then
  echo "error: libopenmc was not copied into $SRC/openmc/lib/." >&2
  echo "       The Python API will import but openmc.lib / depletion will fail." >&2
  exit 1
fi

# --- 3. Python API into .venv-sim, WITHOUT re-resolving dependencies -------
echo ">>> installing OpenMC Python API into $VENV"
UV_PROJECT_ENVIRONMENT="$VENV" uv pip install --no-deps --python "$VENV" "$SRC"

# --- 4. verify -------------------------------------------------------------
UV_PROJECT_ENVIRONMENT="$VENV" uv run --no-project --python "$VENV" -- python - <<'PY'
import openmc, openmc.deplete, openmc.lib
print(f"OK  openmc {openmc.__version__}")
print(f"    package : {openmc.__file__}")
print(f"    libopenmc loaded: {openmc.lib._dll._name if hasattr(openmc.lib, '_dll') else 'n/a'}")
PY

cat <<EOF

Done. Remember the nuclear data (not installed by this script):
  export OPENMC_CROSS_SECTIONS=$REPO_ROOT/data/cross_sections.xml
(\`common.setup_paths\` also sets this itself, relative to the data_generation
 directory — so the generation entry points work as-is.)
EOF

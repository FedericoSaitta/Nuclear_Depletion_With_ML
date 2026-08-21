#!/usr/bin/env bash
# One-time setup on the cluster. Idempotent; safe to re-run.
#
# Deliberately does NOT use `module load python/...` or conda: uv provisions
# CPython 3.12 itself from pyproject.toml's `requires-python`. The only site
# modules needed are the compilers/HDF5 for the OpenMC build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- uv itself (user-local, no admin) --------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install            # honours requires-python → CPython 3.12

# --- ML environment --------------------------------------------------------
echo ">>> .venv        (ML: torch + lightning)"
uv sync --extra ml --no-dev --locked

# --- simulation environment ------------------------------------------------
echo ">>> .venv-sim    (OpenMC)"
UV_PROJECT_ENVIRONMENT=.venv-sim uv sync --extra sim --no-dev --locked

if [ "${SKIP_OPENMC_BUILD:-0}" != "1" ]; then
  # site-specific — adjust module names for your cluster
  module load gcc cmake hdf5 2>/dev/null || \
    echo "warning: could not module-load gcc/cmake/hdf5; assuming they are on PATH"
  ./scripts/install_openmc.sh
fi

cat <<'EOF'

Ready.
  training : uv run nucml train --config configs/node.yaml --data datasets/<file>.h5
  datagen  : UV_PROJECT_ENVIRONMENT=.venv-sim uv run python data_generation/datagen.py \
                 --config data_generation/configs/casl_pincell.yaml -n 4 -c 16
EOF

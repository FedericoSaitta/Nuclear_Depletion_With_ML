# Nuclear_Transport_With_ML

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python itself is provisioned by uv — you do
not need to install or `module load` anything.

### Train / evaluate a surrogate (no OpenMC, no nuclear data)

```bash
git clone https://github.com/FedericoSaitta/Nuclear_Transport_With_ML.git
cd Nuclear_Transport_With_ML
uv sync --extra ml
uv run nucml --config ML/main_config.yaml
```

Override any config key from the command line:

```bash
uv run nucml --config ML/main_config.yaml runtime.model=DNN train.num_epochs=5
```

### Generate data with OpenMC (Linux / WSL2 / macOS)

OpenMC is a compiled C++ code and is **not on PyPI**, so it lives in a second
environment and is built from source:

```bash
sudo apt install -y g++ cmake libhdf5-dev        # or `module load gcc cmake hdf5`
UV_PROJECT_ENVIRONMENT=.venv-sim uv sync --extra sim
./scripts/install_openmc.sh                       # builds OpenMC v0.15.2

UV_PROJECT_ENVIRONMENT=.venv-sim uv run python data_generation/datagen.py -n 4 -c 16
```

#### On a Windows laptop: use WSL2

There is no Windows build path for OpenMC — conda-forge and the Docker images
are Linux-only too, and Docker Desktop runs them inside WSL2 regardless. So the
two halves split across the two OSes, sharing one checkout:

| | runs where | environment |
|---|---|---|
| Training (`ML/**`) | Windows, on the GPU | `.venv` |
| Datagen (`data_generation/**`) | WSL2 Ubuntu | `.venv-sim` |

WSL sees the repo at `/mnt/c/...`, so datagen output and `data/` are shared with
Windows — no copying between the halves. A venv is **not** portable across the
boundary, though: `.venv` has `Scripts\python.exe` and is unusable from Linux,
which is why each side runs its own `uv`.

Keep the two terminals in their lanes. `--extra sim` in PowerShell does not fail
loudly — it syncs sim packages into the ML `.venv` and then dies on
`ModuleNotFoundError: No module named 'openmc'`, because there is no Windows
OpenMC to find. Recover with `uv sync --extra ml`. Note also that `uv run` picks
its own environment: activating `.venv\Scripts\Activate.ps1` first is unnecessary
and only creates a way for the two to disagree.

One-time setup, from a PowerShell prompt:

```bash
wsl
sudo apt update && sudo apt install -y g++ cmake libhdf5-dev git
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

cd /mnt/c/Users/<you>/path/to/Nuclear_Transport_With_ML
echo 'export UV_PROJECT_ENVIRONMENT=$HOME/venvs/nuc-sim' >> ~/.bashrc
source ~/.bashrc
uv sync --extra sim
OPENMC_SRC=$HOME/openmc-src OPENMC_TARGET_VENV=$HOME/venvs/nuc-sim \
    ./scripts/install_openmc.sh
```

**Keep both the venv and the OpenMC build on ext4, not under `/mnt/c`.** uv's
cache lives in the Linux home, so a venv on the Windows drive is a different
filesystem: uv cannot hardlink into it and falls back to copying ~30k files over
the 9p bridge. That is slow and, worse, it can fail *silently* partway — leaving
packages with their `.dist-info` written but no payload. uv then reads that
metadata, believes the package is installed, and never repairs it. The symptom is
a `ModuleNotFoundError: No module named 'numpy'` that survives any number of
`uv sync` runs. Watch for this warning; it means the venv is in the wrong place:

```
warning: Failed to hardlink files; falling back to full copy.
```

Nothing is lost by moving the venv out of the tree — a venv is not portable
across the WSL boundary anyway (`.venv` holds `Scripts\python.exe`). Only the
repo and `data/` need to be shared, and `data/` is fine on `/mnt/c` because it is
read once at worker startup rather than hammered.

Smoke-test before anything long, then every later session is just two commands:

```bash
uv run --extra sim python data_generation/datagen.py -n 1 -c 1     # smoke test
uv run --extra sim python data_generation/datagen.py -n 8 -c 3     # real run
```

Pass `--extra sim` every time. `uv run` re-syncs the environment first, and
without the flag it syncs to the *default* dependency set — quietly uninstalling
scipy, lxml, endf and the rest of the sim extra.

WSL persists across reboots, so the apt packages, `.venv-sim` and the OpenMC
build all survive. Rebuild only when bumping `OPENMC_VERSION`; re-run `uv sync`
only when `pyproject.toml` / `uv.lock` change (it is idempotent and cheap).

**Sizing.** Every datagen worker is a separate process holding its own in-memory
nuclide data, so cores are not the binding constraint — RAM is. On a 16-thread /
16 GB laptop, start at `-c 2` and keep `-c × -t` at 12 or below. WSL2 also claims
~50% of host RAM by default; set `memory=12GB` in `%USERPROFILE%\.wslconfig` to
make that explicit.

Nuclear data (7 GB cross sections + 30 MB depletion chains) is a separate download from
<https://openmc.org/official-data-libraries/> and belongs in `data/`.

### Tests

```bash
uv run pytest              # ML env: OpenMC tests skip automatically
UV_PROJECT_ENVIRONMENT=.venv-sim uv run pytest -m openmc
```



## Downloading Data
To download the cross section data (7 Gb) and Depletion chains (30 Mb) can be done here: https://openmc.org/official-data-libraries/.
The cross section data contains an .xml file along with three folders: Neutron, Photon and wmp. The depletion data is a single .xlm file.

## 6. Everyday commands

| Task | Command |
|---|---|
| Set up ML env | `uv sync --extra ml` |
| Set up sim env | `UV_PROJECT_ENVIRONMENT=.venv-sim uv sync --extra sim` then `./scripts/install_openmc.sh` |
| Set up sim env on Windows | inside `wsl`, see [On a Windows laptop](#on-a-windows-laptop-use-wsl2) |
| Train | `uv run nucml --config ML/main_config.yaml` |
| Train with overrides | `uv run nucml --config ML/main_config.yaml train.num_epochs=5 runtime.device=cpu` |
| Run datagen | `UV_PROJECT_ENVIRONMENT=.venv-sim uv run python data_generation/datagen.py -n 4 -c 16` |
| Add an ML dependency | `uv add --optional ml <pkg>` (updates `pyproject.toml` **and** `uv.lock`) |
| Add a sim dependency | `uv add --optional sim <pkg>` |
| Refresh the lock | `uv lock` |
| Check the lock is current | `uv lock --check` |
| Reproduce exactly | `uv sync --extra ml --locked` (fails if the lock drifted) |
| Export for a `pip`-only machine | `uv export --extra ml --no-hashes -o requirements-ml.txt` |

Bump `UV_PROJECT_ENVIRONMENT=.venv-sim` into your shell profile (or a direnv `.envrc`) if you
find yourself doing datagen work for a whole session.

**PowerShell equivalent** of the env-var prefix:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-sim"; uv sync --extra sim
$env:UV_PROJECT_ENVIRONMENT = $null      # back to .venv
```

(This only helps for `uv sync` bookkeeping on Windows — the OpenMC build and
`datagen.py` itself still have to run inside WSL.)

## Line endings

`.gitattributes` pins `*.sh` to LF and `*.ps1` to CRLF. Windows clones default to
`core.autocrlf=true`, which would otherwise rewrite the shell scripts on checkout
and make bash fail with `$'\r': command not found` under WSL and on the cluster.
Nothing to configure — but if you have a clone predating that file, run
`git add --renormalize .` once.

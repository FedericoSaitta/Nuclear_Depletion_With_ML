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

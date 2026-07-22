#!/bin/bash
set -ex
cd "$(dirname "$0")/"

# This is part of a hack to get dependencies needed for the TTS Rust server, because it integrates a Python component
[ -f pyproject.toml ] || curl -sfLO https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml
[ -f uv.lock ] || curl -sfLO https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock

if [ "$(uname)" == "Darwin" ]; then
  # xformers is not supported/needed on macOS
  sed -i '' '/xformers/d' pyproject.toml
fi

# Ensure huggingface_hub is explicitly in the dependencies list to avoid ModuleNotFoundError
python3 -c "
with open('pyproject.toml', 'r') as f:
    t = f.read()
if 'huggingface_hub' not in t:
    t = t.replace('\"torchaudio\",', '\"torchaudio\",\n   \"huggingface_hub\",')
    with open('pyproject.toml', 'w') as f:
        f.write(t)
"

[ -d .venv ] || uv venv
source .venv/bin/activate
# Point embedded Python in moshi-server to the canonical Python home and virtual environment package site
export PYTHONHOME=$(python -c 'import sys; print(sys.base_prefix)')
export PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])')

cd ..

# This env var must be set to get the correct environment for the Rust build.
# Must be set before running `cargo install`!
# If you don't have it, you'll see an error like `no module named 'huggingface_hub'`
# or similar, which means you don't have the necessary Python packages installed.
# A fix for building Sentencepiece on GCC 15, see: https://github.com/google/sentencepiece/issues/1108
export CXXFLAGS="-include cstdint"
# Fix compatibility errors with modern CMake versions during audiopus_sys compilation
export CMAKE_POLICY_VERSION_MINIMUM=3.5

if [ "$(uname)" == "Darwin" ]; then
  export DYLD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
  cargo install --features metal moshi-server@0.6.4
else
  export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
  cargo install --features cuda moshi-server@0.6.4
fi

if [ "$(uname)" == "Darwin" ]; then
  uv run --project ./dockerless moshi-server worker --config services/moshi-server/configs/tts.toml --port 8089
else
  uv run --locked --project ./dockerless moshi-server worker --config services/moshi-server/configs/tts.toml --port 8089
fi

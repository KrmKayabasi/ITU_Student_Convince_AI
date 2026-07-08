#!/bin/bash
set -ex
cd "$(dirname "$0")/.."


# We need libpython because the TTS uses a Python component. STT and TTS have the same executable, so we need
# to have libpython even if we don't end up using it. For simplicity, we use the same code as for TTS, even though
# you don't need to install any of these Python packages if you're only using the STT.
[ -d .venv ] || uv venv
source .venv/bin/activate
# Point embedded Python in moshi-server to the canonical Python home and virtual environment package site
export PYTHONHOME=$(python -c 'import sys; print(sys.base_prefix)')
export PYTHONPATH=$(python -c 'import site; print(site.getsitepackages()[0])')
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
moshi-server worker --config services/moshi-server/configs/stt.toml --port 8090

#!/usr/bin/env bash
set -euo pipefail

export PATH="/nfs/home/proj2_env/cmake/bin:$PATH"
export PATH="/usr/local/cuda/bin:$PATH"
export CUDACXX="/usr/local/cuda/bin/nvcc"

rm -rf build
cmake -S . -B build
cmake --build build -j 4


# #pip install cmake

# rm -r build
# mkdir build
# cd build
# # cmake -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ ..
# cmake ..
# make -j 4

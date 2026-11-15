#!/usr/bin/env bash
set -euo pipefail

ROOT_PWD="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${ROOT_PWD}/build/build_linux_aarch64"
CACHE_FILE="${BUILD_DIR}/CMakeCache.txt"
# A copied project can contain a CMake cache with absolute paths from the old machine.
if [[ -f "${CACHE_FILE}" ]]; then
    cached_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CACHE_FILE}" | head -n 1)"
    if [[ "${cached_source}" != "${ROOT_PWD}" ]]; then
        archive_root="${XDG_CACHE_HOME:-${HOME}/.cache}/easyrknn_build_archive"
        mkdir -p "${archive_root}"
        archive_path="${archive_root}/build_linux_aarch64_$(date +%Y%m%d_%H%M%S)_$$"
        mv "${BUILD_DIR}" "${archive_path}"
        echo "Moved stale CMake build cache to: ${archive_path}"
    fi
fi
mkdir -p "${BUILD_DIR}"

# Native build on RK3588/aarch64 is preferred. Cross compile when running on x86_64.
ARCH="$(uname -m)"
CMAKE_ARGS=()
if [[ "${ARCH}" != "aarch64" && "${ARCH}" != "arm64" ]]; then
    GCC_COMPILER="${GCC_COMPILER:-aarch64-linux-gnu}"
    export CC="${GCC_COMPILER}-gcc"
    export CXX="${GCC_COMPILER}-g++"
    CMAKE_ARGS+=("-DCMAKE_SYSTEM_NAME=Linux" "-DCMAKE_SYSTEM_PROCESSOR=aarch64")
else
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
fi

cmake -S "${ROOT_PWD}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j"$(nproc)"
cmake --install "${BUILD_DIR}"

echo "Build complete: ${ROOT_PWD}/install/rknn_deploy_Linux"

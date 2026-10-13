#!/bin/sh
set -eu

debug=OFF
timing=OFF
for arg in "$@"; do
    case "$arg" in
        --algo-debug) debug=ON ;;
        --algo-timing) timing=ON ;;
        *) printf 'Usage: sh ./build.sh [--algo-debug] [--algo-timing]\n' >&2; exit 1 ;;
    esac
done

# Native Linux / Raspberry Pi OS build. Requires a C++17 compiler, make, curl and tar.
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tools="$root/.tools"
deps="$root/.deps"
jobs=${JOBS:-$(getconf _NPROCESSORS_ONLN)}
case "$jobs" in
    ''|*[!0-9]*) printf 'JOBS must be a positive integer\n' >&2; exit 1 ;;
esac
[ "$jobs" -gt 0 ] || { printf 'JOBS must be a positive integer\n' >&2; exit 1; }
mkdir -p "$tools" "$deps"

download() {
    url=$1 archive=$2 sha256=$3
    if [ ! -f "$archive" ]; then
        printf 'Downloading %s\n' "$url"
        curl --fail --location --retry 3 --output "$archive.part" "$url"
        mv "$archive.part" "$archive"
    fi
    if [ -n "$sha256" ]; then
        printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --status
    fi
}

# A Pi can run a 64-bit kernel with 32-bit userland: uname alone is insufficient.
case "$(uname -m)/$(getconf LONG_BIT)" in
    x86_64/64|amd64/64) cmake_name=cmake-4.4.4-linux-x86_64 ;;
    aarch64/64|arm64/64) cmake_name=cmake-4.4.4-linux-aarch64 ;;
    *) cmake_name=cmake-4.4.4 ;; # 32-bit ARM and other CPUs: build CMake natively.
esac
cmake="$tools/$cmake_name/bin/cmake"
if [ ! -x "$cmake" ]; then
    base=https://github.com/Kitware/CMake/releases/download/v4.4.4
    download "$base/cmake-4.4.4-SHA-256.txt" "$tools/cmake-4.4.4-SHA-256.txt" \
        560231a0064d3f39c078d6b242dab4a93a8ed17a80e997a56851f338a91e9d65
    digest=$(awk -v name="$cmake_name.tar.gz" '$2 == name {print $1}' "$tools/cmake-4.4.4-SHA-256.txt")
    [ -n "$digest" ] || { printf 'Missing CMake checksum\n' >&2; exit 1; }
    download "$base/$cmake_name.tar.gz" "$tools/$cmake_name.tar.gz" "$digest"
    if [ "$cmake_name" = cmake-4.4.4 ]; then
        mkdir -p "$tools/cmake-source"
        tar -xf "$tools/$cmake_name.tar.gz" -C "$tools/cmake-source" --strip-components=1
        (cd "$tools/cmake-source" && ./bootstrap --prefix="$tools/$cmake_name" \
            --parallel="$jobs" -- -DCMAKE_USE_OPENSSL=OFF && make -j "$jobs" && make install)
    else
        tar -xf "$tools/$cmake_name.tar.gz" -C "$tools"
    fi
fi

prefix="$deps/opencv"
if [ ! -f "$prefix/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    download https://codeload.github.com/opencv/opencv/tar.gz/refs/tags/4.14.0 \
        "$deps/opencv-4.14.0.tar.gz" ee8fb9b30eb60850431b4656447080e3737b56e45719c92b67f245950609f86e
    if [ ! -f "$deps/opencv-4.14.0/CMakeLists.txt" ]; then
        tar -xf "$deps/opencv-4.14.0.tar.gz" -C "$deps"
    fi
    "$cmake" -S "$deps/opencv-4.14.0" -B "$deps/opencv-build" -G 'Unix Makefiles' \
        -C "$root/cmake/opencv.cmake" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_INSTALL_PREFIX="$prefix"
    "$cmake" --build "$deps/opencv-build" --parallel "$jobs"
    "$cmake" --install "$deps/opencv-build"
fi

"$cmake" -S "$root" -B "$root/build" -G 'Unix Makefiles' \
    -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$prefix/lib/cmake/opencv4" \
    -DCOLORIST_DEBUG="$debug" -DCOLORIST_TIMING="$timing"
"$cmake" --build "$root/build" --parallel "$jobs"

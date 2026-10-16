#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
cmake=${CMAKE:-}
if [[ -z "$cmake" ]]; then
    for candidate in "$root"/.tools/cmake-*/bin/cmake; do
        if [[ -x "$candidate" ]]; then cmake=$candidate; break; fi
    done
fi
cmake=${cmake:-$(command -v cmake || true)}
[[ -n "$cmake" ]] || { echo 'Run build.sh to install local CMake and OpenCV first' >&2; exit 1; }
out=${FILL_OUT:-"$root/build/fill-bench/$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$out"
out=$(cd "$out" && pwd)
"$cmake" -S benchmarks/fill -B build/fill-bench -DCMAKE_BUILD_TYPE=Release \
    '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g' \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DOpenCV_DIR="$root/.deps/opencv/lib/cmake/opencv4"
"$cmake" --build build/fill-bench --parallel "${JOBS:-2}"
{
    date -u
    uname -a
    "$cmake" --version
    if command -v vcgencmd >/dev/null; then vcgencmd get_throttled; vcgencmd measure_clock arm; fi
    sha256sum benchmarks/fill/main.cpp build/fill-bench/fill_bench
    sha256sum test_task_CV_CPP/mask.png test_task_CV_CPP/color_mask.png
} > "$out/environment.txt"
cp build/fill-bench/fill_bench "$out/fill_bench"
cp benchmarks/fill/main.cpp "$out/main.cpp"
cp build/fill-bench/compile_commands.json "$out/compile_commands.json"
# Arguments: [runs=50] [kernels=read,write,roofline] [threads=1,2,3,4]
"$out/fill_bench" test_task_CV_CPP/mask.png test_task_CV_CPP/color_mask.png "$out/results.csv" "$@" \
    | tee "$out/run.log"
if command -v vcgencmd >/dev/null; then vcgencmd get_throttled >> "$out/environment.txt"; fi
echo "Results: $out"

#!/usr/bin/env bash
set -euo pipefail

# Native Linux profiling: bash ./profile.sh [threads=2] [runs=500] [algo=rle_stream]
root=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
threads=${1:-2}
runs=${2:-500}
algo=${3:-rle_stream}
jobs=${JOBS:-2}
for count in "$threads" "$runs" "$jobs"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Threads, runs and JOBS must be positive integers' >&2; exit 1; }
done
[[ $# -le 3 ]] || { echo 'Usage: bash ./profile.sh [threads] [runs] [algo]' >&2; exit 1; }
case "$algo" in
    rle|rle_stream|base_spaghetti|base_sauf|base_bbdt) ;;
    *) echo "Unknown algo: $algo" >&2; exit 1 ;;
esac
command -v perf >/dev/null || { echo 'Install perf: sudo apt install linux-perf' >&2; exit 1; }
[[ -f .deps/opencv/lib/cmake/opencv4/OpenCVConfig.cmake ]] || {
    echo 'Build dependencies first: JOBS=2 sh ./build.sh --algo-timing' >&2; exit 1;
}

cmake=${CMAKE:-}
if [[ -z "$cmake" ]]; then
    for candidate in "$root"/.tools/cmake-*/bin/cmake; do
        if [[ -x "$candidate" ]]; then cmake=$candidate; break; fi
    done
fi
cmake=${cmake:-$(command -v cmake || true)}
[[ -n "$cmake" ]] || { echo 'CMake not found; run build.sh first' >&2; exit 1; }

# Isolated Release build: profiling does not change the normal build's flags.
build="$root/build/perf-build"
out=${PERF_OUT:-"$root/build/perf/$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$out"
out=$(cd "$out" && pwd)
perf_cmd=(perf)
if [[ $EUID -ne 0 ]]; then sudo -v; perf_cmd=(sudo perf); fi
finish() {
    if [[ $EUID -ne 0 ]]; then sudo -n chown -R -- "$(id -u):$(id -g)" "$out" || true; fi
    echo "Results: $out"
}
trap finish EXIT

echo 'Building Release with -O3 -g...'
"$cmake" -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
    '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -g' \
    -DOpenCV_DIR="$root/.deps/opencv/lib/cmake/opencv4" \
    -DCOLORIST_DEBUG=OFF -DCOLORIST_TIMING=ON > "$out/build.log" 2>&1
"$cmake" --build "$build" --parallel "$jobs" >> "$out/build.log" 2>&1
cp "$build/colorist" "$out/colorist"
cp sources/RleAlgo.h sources/RleStreamAlgo.h "$out/"
cp "$build/compile_commands.json" "$out/compile_commands.json"
app=("$out/colorist" --mask "$root/test_task_CV_CPP/mask.png"
    --color_mask "$root/test_task_CV_CPP/color_mask.png" --output "$out/result.png"
    --algo-type "$algo" --lifecycle reuse)
if [[ "$threads" -gt 1 ]]; then app+=(--parallel --threads "$threads"); fi

{
    date -u
    uname -a
    if [[ -f /proc/device-tree/model ]]; then tr -d '\0' < /proc/device-tree/model; echo; fi
    perf --version
    printf 'Algorithm: %s\nThreads: %s\nMeasured runs: %s\nWarm-up runs: 20\n' "$algo" "$threads" "$runs"
    sha256sum "$out/colorist" sources/RleAlgo.h sources/RleStreamAlgo.h test_task_CV_CPP/{mask,color_mask}.png
    if command -v vcgencmd >/dev/null; then vcgencmd get_throttled; vcgencmd measure_clock arm; fi
} > "$out/environment.txt"
perf list > "$out/events.txt" 2>&1

echo 'Measuring 100 iterations without profiling...'
"${app[@]}" --warmup 20 --runs 100 --metrics "$out/baseline.json" > "$out/baseline.log"

event=cycles:u
if ! "${perf_cmd[@]}" stat -e "$event" -- true 2> "$out/pmu-check.txt"; then
    event=cpu-clock:u
    echo 'Hardware cycles unavailable: sampling cpu-clock instead.'
fi
printf 'Sampling event: %s\n' "$event" >> "$out/environment.txt"
echo "Recording $runs iterations ($event)..."
"${perf_cmd[@]}" record -e "$event" -F 499 --call-graph dwarf \
    -o "$out/rle.data" -- "${app[@]}" --warmup 20 --runs "$runs" \
    --metrics "$out/recorded.json" > "$out/record.log" 2> "$out/record-perf.log"

echo 'Exporting hotspots, thread breakdown and annotated assembly...'
"${perf_cmd[@]}" report -i "$out/rle.data" --stdio --no-children --call-graph none --sort symbol \
    > "$out/hotspots.txt" 2> "$out/report-errors.txt"
"${perf_cmd[@]}" report -i "$out/rle.data" --stdio --no-children --call-graph none --sort comm,pid,symbol \
    > "$out/threads.txt" 2>> "$out/report-errors.txt"
[[ -s "$out/hotspots.txt" && -s "$out/threads.txt" ]] || { echo 'Perf report export is empty' >&2; exit 1; }
"${perf_cmd[@]}" annotate -i "$out/rle.data" --stdio --source \
    > "$out/assembly.txt" 2> "$out/annotate-errors.txt"

echo 'Collecting hardware counters (three repetitions)...'
if ! "${perf_cmd[@]}" stat -r 3 \
    -e cycles:u,instructions:u,cache-references:u,cache-misses:u,branches:u,branch-misses:u \
    -o "$out/counters.txt" -- "${app[@]}" --warmup 20 --runs "$runs" \
    --metrics "$out/stat.json" > "$out/stat-run.log" 2> "$out/stat-errors.txt"; then
    echo 'Hardware counters failed; collecting software counters instead.'
    mv "$out/counters.txt" "$out/counters-hardware-failed.txt"
    "${perf_cmd[@]}" stat -r 3 -e task-clock,context-switches,cpu-migrations,page-faults \
        -o "$out/counters.txt" -- "${app[@]}" --warmup 20 --runs "$runs" \
        --metrics "$out/stat.json" > "$out/stat-run.log" 2>> "$out/stat-errors.txt"
fi
if command -v vcgencmd >/dev/null; then vcgencmd get_throttled >> "$out/environment.txt"; fi
echo 'Profile complete. Baseline timings exclude image I/O; perf profiles cover the whole process.'

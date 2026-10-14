# Raspberry Pi: build and benchmark

Build **on the Pi**, from the project root. For Pi 3/4/5, prefer
[64-bit Raspberry Pi OS](https://www.raspberrypi.com/documentation/computers/os.html).
32-bit OS is also supported; its first build additionally compiles CMake.
Copy/clone the sources and `test_task_CV_CPP/*.png`, without the Windows
`build/`, `.tools/`, `.deps/` or `.venv/` directories.

## Build

```sh
sudo apt update
sudo apt install -y build-essential curl ca-certificates python3 python3-venv
JOBS=2 sh ./build.sh --algo-timing
```

This builds Release C++17 with diagnostics disabled and stage timing enabled.
CMake 4.4.4 and OpenCV 4.14.0 stay inside the project. OpenCV uses its pthread
parallel backend; CMake selects NEON when supported. Check the configuration's
`colorist SIMD:` message. `JOBS=2` limits compilation jobs to reduce RAM usage;
use `JOBS=1` if compilation runs out of memory. Rerun the build after editing code.

## Run and create a reference

```sh
./build/colorist --mask test_task_CV_CPP/mask.png --color_mask test_task_CV_CPP/color_mask.png --output build/reference.png --algo-type base_spaghetti
./build/colorist --mask test_task_CV_CPP/mask.png --color_mask test_task_CV_CPP/color_mask.png --output build/result-rle.png --algo-type rle --parallel --threads 4
```

Omit `--parallel --threads 4` for sequential execution. `--parallel` alone
uses OpenCV's default thread count; `--threads 4` selects four execution threads.
Compilation `JOBS` does not affect this setting.

## Benchmark all algorithms and compare outputs

Use a local Python environment for Pillow, which enables reference comparison:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-bench.txt
.venv/bin/python bench.py \
    --mask test_task_CV_CPP/mask.png \
    --color-mask test_task_CV_CPP/color_mask.png \
    --algos base_spaghetti base_sauf base_bbdt rle \
    --threads 4 --warmup 3 --runs 30 \
    --reference build/reference.png
```

The default matrix includes sequential/parallel and fresh/reuse configurations,
with shuffled execution. BBDT uses SAUF for our 4-way connectivity. Each table
reports median, mean, standard deviation, P95, min/max and reference PASS/FAIL.
Image I/O is excluded. Results and JSON/CSV reports go into `build/bench/`.
Change `--threads` to compare thread counts; add `--lifecycles reuse` to measure
only buffer reuse. Keep cooling and background workload consistent between runs.

`rle` uses definite component completion. Finished interior components are
painted once during scanning, and their histogram slots are recycled. Components
touching internal stripe edges wait for reconciliation. Input remains BGRA and
output is BGR. Linked run metadata stays outside output pixels.

`Scan/CCL/hist/paint` includes early selection/painting and completion bookkeeping;
final coloring only paints deferred boundary components. Judge total time.
Copy updated `sources/RleAlgo.h`, `sources/main.cpp` and `bench.py` and rebuild.

```sh
.venv/bin/python bench.py --mask test_task_CV_CPP/mask.png --color-mask test_task_CV_CPP/color_mask.png --algos rle --threads 2 --warmup 3 --runs 30 --reference build/reference.png --output-dir build/bench-rle
```

## Profile with perf

```sh
sudo apt install -y linux-perf
bash ./profile.sh 2 500
```

Arguments select execution threads and measured iterations (defaults: 2 and 500).
The script builds an isolated Release executable with `-O3 -g`, then records
samples, exports hotspots/annotated assembly and collects counters. It also saves
100 unprofiled measurements for timing comparisons. Results, source/executable
snapshots and metadata go into a timestamped directory under `build/perf/`.
Run `bash ./profile.sh 1 500` for sequential profiling. Perf covers the whole
process; use the unprofiled measurements for algorithm time. `sudo` is used for
perf access without changing system profiling permissions.

## Compare buffer fills

After the normal build, run:

```sh
bash ./bench_fill.sh 30
```

This standalone benchmark compares `std::fill`, SIMD stores, SIMD stores with
an alignment prefix, and `Mat::setTo`, using one/two threads. It tests small/full
buffers, shifted addresses and foreground runs from the input mask. Each case
verifies its pixels and buffer guards. CSV timings and source/executable snapshots
go into `build/fill-bench/<timestamp>/`. Input loading and optional scan-like
preparation are outside fill timing; this measures writes, not the whole algorithm.

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

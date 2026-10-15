# Raspberry Pi: build, benchmark, profile

Build on the Pi from the project root (64-bit Raspberry Pi OS recommended).
Copy the sources and `test_task_CV_CPP/*.png`, without `build/`, `.tools/`, `.deps/` or `.venv/`.

## Build

```sh
sudo apt update
sudo apt install -y build-essential curl ca-certificates python3 python3-venv
JOBS=2 sh ./build.sh --algo-timing
```

Release C++17 with per-stage timing. CMake and OpenCV are built inside the project;
the configure step prints `colorist SIMD: NEON`. `JOBS=2` limits compile jobs to save
RAM (use `JOBS=1` if the compiler runs out of memory). Rebuild after editing sources.

## Run

```sh
./build/colorist --mask test_task_CV_CPP/mask.png --color_mask test_task_CV_CPP/color_mask.png \
                 --output build/reference.png --algo-type base_spaghetti
./build/colorist --mask test_task_CV_CPP/mask.png --color_mask test_task_CV_CPP/color_mask.png \
                 --output build/result.png --algo-type rle --parallel --threads 2
```

## Benchmark with output comparison

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-bench.txt
.venv/bin/python bench.py --mask test_task_CV_CPP/mask.png --color-mask test_task_CV_CPP/color_mask.png \
    --algos base_spaghetti rle --modes parallel --threads 2 --warmup 10 --runs 100 \
    --reference build/reference.png
```

Results and JSON/CSV reports go to `build/bench/`. Keep cooling and background load
the same between runs.

## Profile

```sh
sudo apt install -y linux-perf
bash ./profile.sh 2 500        # [threads] [runs] [algo=rle]
bash ./perf_scan.sh            # [algos...], default rle
```

- `profile.sh` builds a separate `-O3 -g` binary, records `perf` samples and exports
  hotspots and annotated assembly to `build/perf/<timestamp>/`.
- `perf_scan.sh` runs 1–4 threads and reports per-frame cycles, instructions, IPC,
  stalls, branch misses and DRAM reads (`L3 refill` × 64 bytes). Each configuration
  runs with 20 and 220 frames; the difference removes image loading and saving.
  Results go to `build/perf-scan/<timestamp>/summary.txt`.

Both use `sudo perf`; system profiling permissions are not changed.

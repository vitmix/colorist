#!/bin/bash
# Per-frame hardware counters vs thread count, for answering "why doesn't it scale".
# Usage (from the project root on the Pi, after `JOBS=2 sh ./build.sh --algo-timing`):
#   bash ./perf_scan.sh [algos...]          default: rle
# Each config runs with 20 and 220 measured frames; the difference / 200 removes
# image load/save and process startup, leaving pure Compute() cost per frame.
set -euo pipefail

ALGOS=("$@")
[ $# -eq 0 ] && ALGOS=(rle)
THREADS=(1 2 3 4)
SHORT=20
LONG=220
EXE=./build/colorist
OUT="build/perf-scan/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT"

[ -x "$EXE" ] || { echo "Build first: JOBS=2 sh ./build.sh --algo-timing" >&2; exit 1; }

# Two groups of <= 6 events: the Cortex-A76 has 6 programmable counters, so
# staying within one group avoids multiplexing and scaling estimates.
GROUP_A=(cycles instructions stall_backend branch-misses l1d_cache_refill l2d_cache_refill)
GROUP_B=(l3d_cache_refill ll_cache_miss_rd bus_access_rd bus_access_wr mem_access stall_backend_mem)

supported() {  # Keep only events this kernel/PMU accepts.
    local kept=() e
    for e in "$@"; do
        if sudo perf stat -e "$e" -- true >/dev/null 2>&1; then kept+=("$e")
        else echo "  (event not supported, skipped: $e)" >&2; fi
    done
    local IFS=,; echo "${kept[*]}"
}
echo "Checking PMU events..."
EV_A=$(supported "${GROUP_A[@]}")
EV_B=$(supported "${GROUP_B[@]}")
echo "group A: $EV_A"; echo "group B: $EV_B"

{
    date -u; uname -a; cat /proc/device-tree/model 2>/dev/null | tr -d '\0'; echo
    vcgencmd get_throttled 2>/dev/null || true
    sha256sum "$EXE" sources/RleAlgo.h
} > "$OUT/environment.txt"

run() {  # algo threads frames group_name events
    local algo=$1 t=$2 n=$3 g=$4 ev=$5
    local tag="${algo}_t${t}_n${n}_${g}"
    sudo perf stat -x, -r 3 -e "$ev" -o "$OUT/$tag.csv" -- \
        "$EXE" --mask test_task_CV_CPP/mask.png --color_mask test_task_CV_CPP/color_mask.png \
        --output "$OUT/result.png" --algo-type "$algo" --lifecycle reuse \
        --parallel --threads "$t" --warmup 0 --runs "$n" > "$OUT/$tag.log"
}

for algo in "${ALGOS[@]}"; do
    for t in "${THREADS[@]}"; do
        echo "== $algo, $t thread(s)"
        # Sample the ARM clock while this configuration runs.
        ( while :; do vcgencmd measure_clock arm 2>/dev/null; sleep 0.2; done ) > "$OUT/clock_${algo}_t${t}.txt" &
        clock_pid=$!
        for n in $SHORT $LONG; do
            run "$algo" "$t" "$n" A "$EV_A"
            run "$algo" "$t" "$n" B "$EV_B"
        done
        kill "$clock_pid" 2>/dev/null || true; wait "$clock_pid" 2>/dev/null || true
    done
done
vcgencmd get_throttled 2>/dev/null >> "$OUT/environment.txt" || true

python3 - "$OUT" "$SHORT" "$LONG" "${ALGOS[*]}" "${THREADS[*]}" <<'EOF' | tee "$OUT/summary.txt"
import sys, re, pathlib
out, short, long_, algos, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(), sys.argv[5].split()
out = pathlib.Path(out); frames = long_ - short

def counters(path):
    vals = {}
    for line in path.read_text().splitlines():
        f = line.split(",")
        if len(f) < 3 or line.startswith("#"):
            continue
        try: vals[f[2].split(":")[0]] = float(f[0])
        except ValueError: pass  # <not supported> / <not counted>
    return vals

def wall(path):
    m = re.findall(r"Algorithm time \(mean\): ([\d.]+)", path.read_text())
    return float(m[-1]) if m else float("nan")

def clock(path):
    hz = [int(x) for x in re.findall(r"=(\d+)", path.read_text())]
    return (min(hz) / 1e6, max(hz) / 1e6) if hz else (float("nan"),) * 2

rows = []
for a in algos:
    for t in threads:
        per = {}
        for g in "AB":
            s = counters(out / f"{a}_t{t}_n{short}_{g}.csv")
            l = counters(out / f"{a}_t{t}_n{long_}_{g}.csv")
            for k in l:
                if k in s: per[k] = (l[k] - s[k]) / frames
        rows.append((a, t, per, wall(out / f"{a}_t{t}_n{long_}_A.log"), clock(out / f"clock_{a}_t{t}.txt")))

M = 1e6
print(f"Per-frame values (difference of {long_} and {short} frames, perf -r 3 averages)\n")
print(f"{'algo':<11}{'thr':>4}{'ms':>8}{'MHz min-max':>14}{'Gcyc':>7}{'Minstr':>8}{'IPC':>6}"
      f"{'stall_be%':>10}{'stall_mem%':>11}{'L2 refill M':>12}{'L3 refill M':>12}{'L3 MB':>7}{'rd MB/s':>9}")
for a, t, p, ms, (cmin, cmax) in rows:
    cyc, ins = p.get("cycles", float("nan")), p.get("instructions", float("nan"))
    l3 = p.get("l3d_cache_refill", p.get("ll_cache_miss_rd", float("nan")))
    l3mb = l3 * 64 / M
    print(f"{a:<11}{t:>4}{ms:>8.2f}{f'{cmin:.0f}-{cmax:.0f}':>14}{cyc/1e9:>7.3f}{ins/M:>8.1f}{ins/cyc:>6.2f}"
          f"{100*p.get('stall_backend', float('nan'))/cyc:>10.1f}{100*p.get('stall_backend_mem', float('nan'))/cyc:>11.1f}"
          f"{p.get('l2d_cache_refill', float('nan'))/M:>12.2f}{l3/M:>12.2f}{l3mb:>7.1f}{l3mb/ms*1000:>9.0f}")
print("\nRaw per-frame counters:")
for a, t, p, ms, _ in rows:
    print(f"{a} t={t}: " + ", ".join(f"{k}={v/M:.2f}M" for k, v in sorted(p.items())))
EOF
echo "Results: $OUT"

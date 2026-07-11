#!/bin/bash
set -e
BIN="./main_flat_pthread"
[ ! -f "$BIN" ] && echo "Compile: g++ main_flat_pthread.cc -o main_flat_pthread -O2 -lpthread -std=c++17 -I.." && exit 1

echo "=== [1/4] perf stat ==="
perf stat -e cpu-cycles,instructions,cache-references,cache-misses \
         -e L1-dcache-loads,L1-dcache-load-misses \
         -e branch-misses,context-switches,cpu-migrations \
         -o perf_flat_stat.txt -- $BIN
cat perf_flat_stat.txt

echo "=== [2/4] perf record ==="
perf record -g --call-graph dwarf -e cpu-cycles:pp -o perf_flat_record.data -- $BIN

echo "=== [3/4] perf report ==="
perf report -i perf_flat_record.data --stdio --sort=symbol --percent-limit=1.0 > perf_flat_report.txt
head -40 perf_flat_report.txt

echo "=== [4/4] perf annotate ==="
for f in "flat_intra_worker" "flat_batch_worker" "InnerProductSIMDNeon"; do
    echo "--- $f ---"
    perf annotate -i perf_flat_record.data --stdio --symbol="$f" 2>/dev/null | head -60 || echo "(not found)"
done > perf_flat_annotate.txt
echo "Done."

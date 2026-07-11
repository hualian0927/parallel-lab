#!/bin/bash
set -e
BIN="./main_hnsw_pthread"
[ ! -f "$BIN" ] && echo "Compile: g++ main_hnsw_pthread.cc -o main_hnsw_pthread -O2 -lpthread -std=c++17 -I.." && exit 1
echo "=== [1/4] perf stat ==="
perf stat -e cpu-cycles,instructions,cache-references,cache-misses \
         -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
         -e branch-instructions,branch-misses,context-switches,cpu-migrations \
         -o perf_hnsw_pthread_stat.txt -- $BIN
cat perf_hnsw_pthread_stat.txt
echo "=== [2/4] perf record ==="
perf record -g --call-graph dwarf -e cpu-cycles:pp -o perf_hnsw_pthread_record.data -- $BIN
echo "=== [3/4] perf report ==="
perf report -i perf_hnsw_pthread_record.data --stdio --sort=symbol --percent-limit=1.0 > perf_hnsw_pthread_report.txt
head -40 perf_hnsw_pthread_report.txt
echo "=== [4/4] perf annotate ==="
for f in "hnswlib::HierarchicalNSW<float>::searchKnn" "hnsw_batch_worker"; do
    echo "--- $f ---"
    perf annotate -i perf_hnsw_pthread_record.data --stdio --symbol="$f" 2>/dev/null | head -80 || echo "(not found)"
done > perf_hnsw_pthread_annotate.txt
echo "Done."

#!/bin/bash

echo "========================================="
echo "Lab1 Complete Performance Benchmark"
echo "Date: $(date)"
echo "========================================="
echo ""

echo "=== 1. Matrix-Vector Test (Different Optimization Levels) ==="
echo ""
echo "--- -O0 (No Optimization) ---"
./matrix_vector_O0
echo ""
echo "--- -O2 (Standard Optimization) ---"
./matrix_vector_O2
echo ""

echo "=== 2. Array Sum Test (Different Unrolling Levels) ==="
echo ""
./sum_array
echo ""

echo "=== 3. Floating Point Precision Experiment ==="
echo ""
./float_precision
echo ""

echo "=== Experiment Complete ==="

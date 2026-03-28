#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define REPEAT 100  // 重复执行次数

// 平凡算法：逐列访问
double* column_major(double* matrix, double* vector, int n) {
    double* result = (double*)malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++) {
        result[i] = 0.0;
        for (int j = 0; j < n; j++) {
            result[i] += matrix[j * n + i] * vector[j];
        }
    }
    return result;
}

// 优化算法：逐行访问
double* row_major(double* matrix, double* vector, int n) {
    double* result = (double*)calloc(n, sizeof(double));
    
    for (int j = 0; j < n; j++) {
        double vj = vector[j];
        for (int i = 0; i < n; i++) {
            result[i] += matrix[j * n + i] * vj;
        }
    }
    return result;
}

// 高精度计时函数
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// 测试性能
void test_performance(int n, const char* name, double* (*func)(double*, double*, int)) {
    double* matrix = (double*)malloc(n * n * sizeof(double));
    double* vector = (double*)malloc(n * sizeof(double));
    
    if (!matrix || !vector) {
        printf("Memory allocation failed!\n");
        return;
    }
    
    // 初始化数据
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (double)(i + 1);
    }
    for (int i = 0; i < n; i++) {
        vector[i] = (double)(i + 1);
    }
    
    double total_time = 0;
    for (int r = 0; r < REPEAT; r++) {
        double start = get_time();
        double* result = func(matrix, vector, n);
        double end = get_time();
        total_time += (end - start);
        free(result);
    }
    
    printf("%-20s n=%-6d average time: %.8f seconds\n", 
           name, n, total_time / REPEAT);
    
    free(matrix);
    free(vector);
}

int main() {
    printf("\n========================================\n");
    printf("Matrix-Vector Dot Product Performance Test\n");
    printf("Repeat times: %d for each size\n", REPEAT);
    printf("========================================\n\n");
    
    int sizes[] = {100, 200, 500, 1000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        printf("--- Matrix Size: %d x %d ---\n", sizes[i], sizes[i]);
        test_performance(sizes[i], "Column Major (naive)", column_major);
        test_performance(sizes[i], "Row Major (optimized)", row_major);
        printf("\n");
    }
    
    return 0;
}
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define REPEAT 1000  // 重复执行次数

// 平凡算法：链式累加
double chain_sum(double* arr, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

// 优化算法1：两路链式累加
double two_way_sum(double* arr, int n) {
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += arr[i];
        if (i + 1 < n) {
            sum2 += arr[i + 1];
        }
    }
    return sum1 + sum2;
}

// 优化算法2：四路链式累加
double four_way_sum(double* arr, int n) {
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    int i;
    for (i = 0; i + 3 < n; i += 4) {
        sum1 += arr[i];
        sum2 += arr[i + 1];
        sum3 += arr[i + 2];
        sum4 += arr[i + 3];
    }
    for (; i < n; i++) {
        sum1 += arr[i];
    }
    return sum1 + sum2 + sum3 + sum4;
}

// 优化算法3：八路链式累加
double eight_way_sum(double* arr, int n) {
    double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
    double sum5 = 0, sum6 = 0, sum7 = 0, sum8 = 0;
    int i;
    for (i = 0; i + 7 < n; i += 8) {
        sum1 += arr[i];
        sum2 += arr[i + 1];
        sum3 += arr[i + 2];
        sum4 += arr[i + 3];
        sum5 += arr[i + 4];
        sum6 += arr[i + 5];
        sum7 += arr[i + 6];
        sum8 += arr[i + 7];
    }
    // 处理剩余元素
    for (; i < n; i++) {
        sum1 += arr[i];
    }
    return sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8;
}

// 高精度计时函数
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// 测试性能
void test_performance(int n, const char* name, double (*func)(double*, int)) {
    double* arr = (double*)malloc(n * sizeof(double));
    
    if (!arr) {
        printf("Memory allocation failed!\n");
        return;
    }
    
    // 初始化数据
    for (int i = 0; i < n; i++) {
        arr[i] = (double)(i + 1);
    }
    
    double total_time = 0;
    double result = 0;
    
    for (int r = 0; r < REPEAT; r++) {
        double start = get_time();
        result = func(arr, n);
        double end = get_time();
        total_time += (end - start);
    }
    
    printf("%-20s n=%-7d average time: %.8f seconds\n", 
           name, n, total_time / REPEAT);
    
    free(arr);
}

int main() {
    printf("\n========================================\n");
    printf("Array Sum Performance Test\n");
    printf("Repeat times: %d for each size\n", REPEAT);
    printf("========================================\n\n");
    
    int sizes[] = {1000, 10000, 100000, 500000, 1000000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int i = 0; i < num_sizes; i++) {
        printf("--- Array Size: %d ---\n", sizes[i]);
        test_performance(sizes[i], "Chain Sum", chain_sum);
        test_performance(sizes[i], "Two-Way Sum", two_way_sum);
        test_performance(sizes[i], "Four-Way Sum", four_way_sum);
        test_performance(sizes[i], "Eight-Way Sum", eight_way_sum);
        printf("\n");
    }
    
    return 0;
}
#include <stdio.h>

int main() {
    float a = 1e20f;      // 很大的数
    float b = 1e-20f;     // 很小的数
    float c = -1e20f;     // 很大的负数
    
    printf("========================================\n");
    printf("Floating Point Precision Experiment\n");
    printf("========================================\n\n");
    
    printf("Values:\n");
    printf("  a = %e\n", a);
    printf("  b = %e\n", b);
    printf("  c = %e\n\n", c);
    
    // 顺序1: (a + b) + c
    float r1 = (a + b) + c;
    printf("Order 1: (a + b) + c = %e\n", r1);
    
    // 顺序2: a + (b + c)
    float r2 = a + (b + c);
    printf("Order 2: a + (b + c) = %e\n", r2);
    
    // 顺序3: (a + c) + b
    float r3 = (a + c) + b;
    printf("Order 3: (a + c) + b = %e\n", r3);
    
    // 顺序4: a + c + b
    float r4 = a + c + b;
    printf("Order 4: a + c + b = %e\n\n", r4);
    
    printf("Analysis:\n");
    printf("  (a+b)+c = %e\n", r1);
    printf("  a+(b+c) = %e\n", r2);
    printf("  Difference = %.0e\n\n", r2 - r1);
    
    printf("Conclusion:\n");
    printf("  Floating-point addition is not associative!\n");
    printf("  The order of operations affects the result due to precision limitations.\n");
    printf("  (a+b)+c results in 0 because b is too small compared to a.\n");
    printf("  a+(b+c) preserves b because (b+c) = -c+b? Actually b+c = b + (-a) gives...\n");
    
    return 0;
}

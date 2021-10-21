#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


int main() {
    int i, count, tid;
    double x, y, z, pi;
    unsigned long n = 1000000;
    unsigned int seed;
    count = 0;

#pragma omp parallel private(x, y, z, tid, seed) reduction(+:count)
    tid = omp_get_thread_num();
    seed = (unsigned) time(NULL);
    seed = (seed & 0xFFFFFFF0) | (tid + 1);
    srand(seed);

#pragma omp for 
    for(i = 0; i < n; ++i) {
        x = ((double)rand_r(&seed) / RAND_MAX) * 2 - 1;
        y = ((double)rand_r(&seed) / RAND_MAX) * 2 - 1;
        z = x * x + y * y;
        if(z <= 1)
            count += 1;
    }
#pragma omp barrier
    pi = (double) count / n * 4;
    printf("Approximate PI = %f\n", pi);
    return(0);
}

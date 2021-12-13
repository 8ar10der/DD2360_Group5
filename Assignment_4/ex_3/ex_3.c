#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>

void acc_saxpy(int n, float a, float *x, float *y) {
    // #pragma acc data
        #pragma acc parallel loop 
        for (int i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
}

void cpu_saxpy(int n, float a, float *x, float *y){
    for (int i = 0; i < n; i++){
        y[i] = a * x[i] + y[i];
    }
} 

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char const *argv[])
{   
    int N = atoi(argv[1]);
    float *x,*c_y,*g_y;
    //localhost pointer
    x = (float*)malloc(N*sizeof(float));
    c_y = (float*)malloc(N*sizeof(float));
    g_y = (float*)malloc(N*sizeof(float));
    //init array
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        c_y[i] = 1.0f;
        g_y[i] = 1.0f;
    }

    const float a = 2.0F;

    //saxpy
    printf("Computing SAXPY on the CPU…  ");
    double CPUTime = cpuSecond();
    cpu_saxpy(N,a,x,c_y);
    CPUTime = cpuSecond() - CPUTime;
    printf("Done! The time of the cpu computing is: %fs\n", CPUTime);
    printf("Computing SAXPY on the GPU…  ");
    double GPUTime = cpuSecond();
    acc_saxpy(N,a,x,g_y);
    GPUTime = cpuSecond() - GPUTime;
    printf("Done! The time of the gpu computing is: %fs\n", GPUTime);
    //comparing
    int errorCount = 0;
    printf("Comparing the output for each implementation…  ");
    for (int i = 0; i < N; i++){
        // printf("<%f,%f>",c_y[i],g_y[i]);
        if (abs(c_y[i]-g_y[i]) >= 0.5f)
            errorCount++;
    }
    if (errorCount == 0){
        printf("Correct!\n");
    } else {
        printf("Not Pass, there are %d differences.\n", errorCount);
    }
    
    //free
    free(g_y);
    free(c_y);
    free(x);
}

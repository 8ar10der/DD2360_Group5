#include<stdio.h>

#define ARRAY_SIZE 10000;


__global__ void gpu_saxpy(float a, float *x, float *y){
    int N = ARRAY_SIZE;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
        y[i] = a * x[i] + y[i];
}

void cpu_saxpy(float a, float *x, float *y){
    int N = ARRAY_SIZE;
    for (int i = 0; i < N; i++){
        y[i] = a * x[i] + y[i];
    }
} 

int main(int argc, char const *argv[])
{
    int N = ARRAY_SIZE;
    float *x,*c_y,*g_y;
    float *d_x,*d_y;
    //localhost pointer
    x = (float*)malloc(N*sizeof(float));
    c_y = (float*)malloc(N*sizeof(float));
    g_y = (float*)malloc(N*sizeof(float));
    //CUDA device pointer
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    //init array
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        c_y[i] = 1.0f;
        g_y[i] = 1.0f;
    }
    cudaMemcpy(d_x,x,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,g_y,N*sizeof(float),cudaMemcpyHostToDevice);
    //saxpy
    printf("Computing SAXPY on the CPU…  ");
    cpu_saxpy(2.0f,x,c_y);
    printf("Done!\n");
    printf("Computing SAXPY on the GPU…  ");
    gpu_saxpy<<<(N+255)/256, 256>>>(2.0f,d_x,d_y);
    printf("Done!\n");
    //copy result
    cudaMemcpy(g_y,d_y,N*sizeof(float),cudaMemcpyDeviceToHost);
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
    cudaFree(d_y);
    cudaFree(d_x);
    free(g_y);
    free(c_y);
    free(x);
}

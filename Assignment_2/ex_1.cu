/* run nvcc -arch-sm_50 code.cu -o -helloworld */

#include <stdio.h>
#define N 1
/* number of böpcls */
#define TPB 256
/* number of threads */
__global__ void printthreadKernel(){
  int threadID = threadIdx.x;
  /* only one block, no need for blockIdx or dim */
  printf("Hello world! My threadID is %d!\n", threadID);
  /*print out hello world and ID nr*/
}

int main(){

  printthreadKernel<<<N, TPB>>>();
  /* one block, 256 threads */
  cudaDeviceSynchronize();
  return (0);
}

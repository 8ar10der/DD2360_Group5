#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>
#include <sys/time.h>

/* define threads and blocks */
#define N 32//blocks
#define TPB 256//threads
#define TRIALS 1000000 //trials per thread


//randomized cointoss
__global__ void cointoss(curandState *states, unsigned int* coinTot, unsigned int* inCirc) {
  //setup thread seed
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int seed = id; //different seed per thread
  // printf("%d\n", *itpb);
  double piValue = -1;
  int trls = TRIALS/(N*TPB); //trials per thread
  curand_init(seed, id, 0, &states[id]); //start CURAND
  for(int i = 0; i < trls; i++) {
    //printf("Line %d, id: %d\n", __LINE__, i);
    double x = curand_uniform(&states[id]);
    double y = curand_uniform(&states[id]);
    //printf("Line %d, id: %d, x: %f, y: %f\n", __LINE__, id, x , y);
    double throwDist = sqrt(x*x + y*y); //check throw distance
    // printf("Line %d, id: %d, throwDist: %f\n", __LINE__, id, throwDist);
    atomicAdd(coinTot, 1); //increase cointot by 1
    //checks if coin is within circle
    if(throwDist <= 1.0){
      atomicAdd(inCirc, 1); //increase inCirc by 1
    }
    piValue = 4 * (double) *inCirc / (double) *coinTot;
    // printf("Line %d, id: %d, throwDist: %f\n", __LINE__, id, throwDist);
    printf("tID: %d, Pi Value %0.10f, iter:%d, cTot:%d \n", id, piValue, i, *coinTot);

  }

}

__host__ double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(){
  unsigned* coinTot; //number of times coin is thrown
  unsigned* inCirc; //number of times coin is in circle
//  int* itpb; //iterations per block

  cudaMalloc(&coinTot, sizeof(unsigned));
  cudaMalloc(&inCirc, sizeof(unsigned));
  //cudaMalloc(&itpb, sizeof(int));

  cudaMemset(coinTot, 0, sizeof(unsigned));
  cudaMemset(inCirc, 0, sizeof(unsigned));
//  printf("helpÖ %d\n", TRIALS/(N*TPB));
  //cudaMemset(itpb, (TRIALS/(N*TPB)), sizeof(int));

  curandState *dev_random;
  cudaMalloc((void**)&dev_random, TPB*sizeof(curandState));
  double iStart = cpuSecond();

  cointoss<<<N, TPB>>>(dev_random, coinTot, inCirc);
  // generates total of in and out of coinTot
  cudaDeviceSynchronize();
  double iElaps = cpuSecond() - iStart;
  int iter = TRIALS;
  printf("Iterations: %d\nTime elapsed %f s\n", iter, iElaps);
  cudaFree(dev_random);

  return 0;
}

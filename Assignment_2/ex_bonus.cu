#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>

/* define threads and blocks */
#define N 32//blocks = N/TPB
#define TPB 512 //threads
#define TRIALS_PER_THREAD 100 //trials per thread

//randomized cointoss
__global__ void cointoss(curandState *states, unsigned int* coinTot, unsigned int* inCirc) {
  //setup thread seed
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  int seed = id; //different seed per thread
  double piValue = -1;
  curand_init(seed, id, 0, &states[id]); //start CURAND
  for(int i = 0; i < TRIALS_PER_THREAD; i++) {
    // printf("Line %d, id: %d\n", __LINE__, id);
    double x = curand_uniform(&states[id]);
    double y = curand_uniform(&states[id]);
    // printf("Line %d, id: %d, x: %f, y: %f\n", __LINE__, id, x , y);

    double throwDist = sqrt(x*x + y*y); //check throw distance
    // printf("Line %d, id: %d, throwDist: %f\n", __LINE__, id, throwDist);
    atomicAdd(coinTot, 1); //increase cointot by 1
    //checks if coin is within circle
    if(throwDist <= 1.0){
      atomicAdd(inCirc, 1); //increase inCirc by 1
    }
    piValue = 4 * (double) *inCirc / (double) *coinTot;
  }
  printf(" Pi Value %0.10f, *inCirc: %d, *coinTot:%d \n", piValue, *inCirc, *coinTot);

}



int main(){
  unsigned* coinTot; //number of times coin is in circle
  unsigned* inCirc; //number of times coin is in circle
  
  cudaMalloc(&coinTot, sizeof(unsigned));
  cudaMalloc(&inCirc, sizeof(unsigned));

  cudaMemset(coinTot, 0, sizeof(unsigned));
  cudaMemset(inCirc, 0, sizeof(unsigned));


  curandState *dev_random;
  cudaMalloc((void**)&dev_random, TPB*sizeof(curandState));

  cointoss<<<TPB/N, TPB>>>(dev_random, coinTot, inCirc);
  // generates total of in and out of coinTot
  cudaDeviceSynchronize();
  cudaFree(dev_random);

  return 0;
}

#include <curand_kernel.h>
#include <curand.h>
#include <stdio.h>

/* define threads and blocks */
#define N //blocks = N/TPB
#define TPB 256 //threads
#define TRIALS_PER_THREAD 10 //trials per thread

curandState *dev_random;
cudaMalloc((void**)&dev_random, NPB*TB*sizeof(curandState));

typedef struct {
  double x,y;
} pos

// create position structure with x and y coordinate

//randomized cointoss
__global__ void cointoss( curandState *states) {
  //setup thread seed
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  float x;
  double inCirc = 0; //number of times coin is in circle
  int seed = id; //different seed per thread

  curand_init(seed, id, 0, &states[id]); //start CURAND
  for(int i = 0; i < TRIALS_PER_THREAD; i++) {
    pos.x = curand_uniform (&states[id]);
    pos.y = curand_uniform (&states[id]);
    throwDist = distance(pos.x, pos.y) //check throw distance

    atomicAdd(coinTot, 1) //increase cointot by 1
    //checks if coin is within circle
    if(throwDist <= 1){
      atomicAdd(inCirc, 1); //increase inCirc by 1
    }
    double pieValue = pieCalc(inCirc, coinTot);
    //calculate pievalue
    printf("Trial %d, Thread %d: Pi Value %d\n", coinTot, id, pieValue);
  }

}

/* distance of coin from origin */
__device__ double distance(double x, double y) {
  double throwDist;
  throwDist = sqrt(x*x + y*y) //pyth. form
  return throwDist
}
/* calculate pi using the coins */
__device__ double pieCalc(int numThrows, double a){
  double pieValue = a/numThrows;
  return pieValue
}



int main(){
  cointoss<<<N/TPB, TPB>>>(inCirc, coinTot);
  // generates total of in and out of coinTot
  cudafree();
  cudaDeviceSynchronize();
  return 0;
}

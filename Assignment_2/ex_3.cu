#include<stdio.h>
#include <cstdlib>
#include <assert.h>
#include <sys/time.h>

#define NUM_PARTICLES 10000000
#define NUM_ITERATIONS 200
#define BLOCKS 128
#define TBP 32 


typedef struct  {
	float3 position;
	float3 velocity;
} Particle;


__global__ void updateParticlesKernel(Particle* particles) {
	// int threadID = threadIdx.x;
	const int i = blockIdx.x  *blockDim.x + threadIdx.x;


    for (unsigned k = 0; k < NUM_ITERATIONS; k++) {
        for (unsigned j = i; j < NUM_PARTICLES; j += TBP*BLOCKS) {
            particles[j].velocity.x +=  0.1;
            particles[j].velocity.y += 0.001;
            particles[j].velocity.z -= 0.002; 
            particles[j].position.x += particles[j].velocity.x * 1;
        }
    }


}

__host__ void checkConsistency(Particle* particlesHost, Particle* particlesDevice, unsigned numberOfParticles){
    for (unsigned i = 0; i < numberOfParticles; i++) {
        #if defined DEBUG
            printf("host: %f, device: %f\n", particlesHost[i].position.x, particlesDevice[i].position.x);
            printf("host: %f, device: %f\n", particlesHost[i].position.y, particlesDevice[i].position.y);
        #endif

        assert (particlesHost[i].position.x == particlesDevice[i].position.x);
        assert (particlesHost[i].position.y == particlesDevice[i].position.y);
        assert (particlesHost[i].position.z == particlesDevice[i].position.z);

        assert (particlesHost[i].velocity.x == particlesDevice[i].velocity.x);
        assert (particlesHost[i].velocity.y == particlesDevice[i].velocity.y);
        assert (particlesHost[i].velocity.z == particlesDevice[i].velocity.z);    
    }
}

__host__ void updateParticles(Particle* particles, unsigned numberOfParticles) {

    for (unsigned k = 0; k < NUM_ITERATIONS; k++) {
        for (unsigned i = 0; i < numberOfParticles; i++) {
            particles[i].velocity.x +=  0.1;
            particles[i].velocity.y += 0.001;
            particles[i].velocity.z -= 0.002; 
            particles[i].position.x += particles[i].velocity.x * 1;
        }
    }


}

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}


__host__ void generateRandomParticles(Particle* particles, unsigned numberOfParticles){
    for (unsigned i = 0; i < numberOfParticles; i++) {
        particles[i].position = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
        particles[i].velocity = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
    }
}

int main(int argc, char const *argv[])
{
    Particle* particles;
    Particle* cudaParticles;


    size_t particlesSize = NUM_PARTICLES*sizeof(Particle);

    //localhost pointer
    particles = (Particle*)malloc(particlesSize);
    cudaMalloc(&cudaParticles, particlesSize);

    Particle* cudaParticlesOnHost;
    cudaParticlesOnHost = (Particle*)malloc(particlesSize);

    generateRandomParticles(particles, NUM_PARTICLES);


    double updateParticlesKernelStart = cpuSecond();
    cudaMemcpy(cudaParticles, particles, particlesSize, cudaMemcpyHostToDevice);
    // <blocks, threads per block> 
    updateParticlesKernel<<<BLOCKS, TBP>>>(cudaParticles);
    cudaDeviceSynchronize();
    cudaMemcpy(cudaParticlesOnHost, cudaParticles, particlesSize, cudaMemcpyDeviceToHost);
    double updateParticlesKernelTime = cpuSecond() - updateParticlesKernelStart;


    double updateParticlesStart = cpuSecond();
    updateParticles(particles, NUM_PARTICLES);
    double updateParticlesTime = cpuSecond() - updateParticlesStart;

    printf("updateParticlesTime: %f updateParticlesKernelTime: %f \n", updateParticlesTime, updateParticlesKernelTime);


    checkConsistency(particles, cudaParticlesOnHost, NUM_PARTICLES);


    free(particles);
    free(cudaParticlesOnHost);
    cudaFree(cudaParticles);
}

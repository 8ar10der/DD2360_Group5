#include<stdio.h>
#include <cstdlib>
#include <assert.h>
#include <sys/time.h>

#define NUM_ITERATIONS 200


typedef struct  {
	float3 position;
	float3 velocity;
} Particle;


__global__ void updateParticlesKernel(Particle* particles, unsigned totalThreads, unsigned totalParticles) {
	const int i = blockIdx.x  *blockDim.x + threadIdx.x;
    
    for (unsigned j = i; j < totalParticles; j += totalThreads) {
        particles[j].velocity.x +=  0.1;
        particles[j].velocity.y += 0.001;
        particles[j].velocity.z -= 0.002; 
        particles[j].position.x += particles[j].velocity.x * 1;
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

    // Number of Particles
    unsigned NUM_PARTICLES = atoi(argv[1]);

    // Number of threads per block
    unsigned TBP = atoi(argv[2]);

    printf("NUM_PARTICLES: %d TBP: %d \n", NUM_PARTICLES, TBP);

    size_t particlesSize = NUM_PARTICLES*sizeof(Particle);

    particles = (Particle*)malloc(particlesSize);
    cudaError_t pinnedMemory = cudaHostAlloc(&cudaParticles, particlesSize, cudaHostAllocDefault);

    if (pinnedMemory != cudaSuccess) {
        printf("pinnedMemory Allocation resulted in error  %d", pinnedMemory);
    }


    generateRandomParticles(particles, NUM_PARTICLES);

    unsigned BLOCKS = (NUM_PARTICLES + TBP - 1)/TBP;
    for (unsigned k = 0; k < NUM_ITERATIONS; k++) { 
        cudaMemcpy(cudaParticles, particles, particlesSize, cudaMemcpyHostToDevice);
        updateParticlesKernel<<<BLOCKS, TBP>>>(cudaParticles, TBP*BLOCKS, NUM_PARTICLES);
        cudaDeviceSynchronize();
        cudaMemcpy(particles, cudaParticles, particlesSize, cudaMemcpyDeviceToHost);
    }

    free(particles);
    cudaFreeHost(cudaParticles);
}
 
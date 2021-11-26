#include <stdio.h>
#include <cstdlib>
#include <assert.h>
#include <sys/time.h>


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

__host__ void generateRandomParticles(Particle* particles, unsigned numberOfParticles){
    for (unsigned i = 0; i < numberOfParticles; i++) {
        particles[i].position = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
        particles[i].velocity = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
    }
}

int main(int argc, char const *argv[])
{
    Particle* particles;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // Number of Particles
    unsigned NUM_PARTICLES = atoi(argv[1]);

    // Number of threads per block
    unsigned TBP = atoi(argv[2]);

    //The length of one batches
    unsigned BATCH_LENGTH = atoi(argv[3]);

    printf("NUM_PARTICLES: %d TBP: %d \n", NUM_PARTICLES, TBP);

    size_t particlesSize = NUM_PARTICLES*sizeof(Particle);
    size_t batchSize = BATCH_LENGTH*sizeof(Particle);

    int size = (NUM_PARTICLES+BATCH_LENGTH-1)/BATCH_LENGTH;
    Particle* batches[size];
    Particle* cudaParticlesBatch[size];

    particles = (Particle*)malloc(particlesSize);
    cudaError_t pinnedMemory = cudaHostAlloc(cudaParticlesBatch, size*sizeof(Particle), cudaHostAllocDefault);

    if (pinnedMemory != cudaSuccess) {
        printf("pinnedMemory Allocation resulted in error  %d", pinnedMemory);
    }

    generateRandomParticles(particles, NUM_PARTICLES);

    //divide particles into batches
    for (int i = 0; i < size; i++){
        batches[i] = particles + i * batchSize;
    }

    unsigned BLOCKS = (NUM_PARTICLES + TBP - 1)/TBP;
    for (unsigned k = 0; k < size; k++) {
        cudaMemcpyAsync(cudaParticlesBatch[k], batches[k], batchSize, cudaMemcpyHostToDevice, stream1);
        updateParticlesKernel<<<BLOCKS, TBP, 0, stream1>>>(cudaParticlesBatch[k], TBP*BLOCKS, BATCH_LENGTH);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(batches[k], cudaParticlesBatch[k], batchSize, cudaMemcpyDeviceToHost, stream1);
    }

    free(particles);
    cudaFreeHost(cudaParticlesBatch);
    cudaStreamDestroy(stream1);
}
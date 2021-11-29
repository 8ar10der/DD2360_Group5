#include <stdio.h>
#include <cstdlib>
#include <assert.h>

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
        printf("host: %f, device: %f\n", particlesHost[i].position.x, particlesDevice[i].position.x);
        printf("host: %f, device: %f\n", particlesHost[i].position.y, particlesDevice[i].position.y);
        printf("host: %f, device: %f\n", particlesHost[i].position.z, particlesDevice[i].position.z);
        printf("host: %f, device: %f\n", particlesHost[i].velocity.x, particlesDevice[i].velocity.x);
        printf("host: %f, device: %f\n", particlesHost[i].velocity.y, particlesDevice[i].velocity.y);
        printf("host: %f, device: %f\n", particlesHost[i].velocity.z, particlesDevice[i].velocity.z);

        assert (particlesHost[i].position.x == particlesDevice[i].position.x);
        assert (particlesHost[i].position.y == particlesDevice[i].position.y);
        assert (particlesHost[i].position.z == particlesDevice[i].position.z);

        assert (particlesHost[i].velocity.x == particlesDevice[i].velocity.x);
        assert (particlesHost[i].velocity.y == particlesDevice[i].velocity.y);
        assert (particlesHost[i].velocity.z == particlesDevice[i].velocity.z);
    }
}

__host__ void updateParticles(Particle* particles, unsigned numberOfParticles) {
        for (unsigned i = 0; i < numberOfParticles; i++) {
            particles[i].velocity.x +=  0.1;
            particles[i].velocity.y += 0.001;
            particles[i].velocity.z -= 0.002;
            particles[i].position.x += particles[i].velocity.x * 1;
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
    // Number of Particles
    unsigned NUM_PARTICLES = atoi(argv[1]);

    // Number of threads per block
    unsigned TBP = atoi(argv[2]);

    //The size of one batches
    unsigned BATCH_LENGTH = atoi(argv[3]);

    unsigned BLOCKS = (NUM_PARTICLES + TBP - 1)/TBP;
    //The number of Streams
    unsigned STREAM_NUM = atoi(argv[4]);

    cudaStream_t streams[STREAM_NUM];
    for(int i =0;i<STREAM_NUM;i++){
        cudaStreamCreate(&streams[i]);
    }
    
    printf("NUM_PARTICLES: %d TBP: %d BL: %d SN: %d \n", NUM_PARTICLES, TBP, BATCH_LENGTH, STREAM_NUM);

    size_t particlesSize = NUM_PARTICLES*sizeof(Particle);
    size_t batchSize = BATCH_LENGTH*sizeof(Particle);
    size_t size = (NUM_PARTICLES+BATCH_LENGTH-1)/BATCH_LENGTH;
    
    Particle* batches[size];
    Particle* cudaParticles;
    Particle* pinnnedParticles;
    
    cudaMalloc(&cudaParticles, particlesSize);
    cudaError_t pinnedMemory = cudaHostAlloc((void**) &pinnnedParticles, particlesSize, cudaHostAllocDefault);

    if (pinnedMemory != cudaSuccess) {
        printf("pinnedMemory Allocation resulted in error  %d", pinnedMemory);
    }

    generateRandomParticles(pinnnedParticles, NUM_PARTICLES);

    //divide particles into batches
    for (int i = 0; i < size; i++){
        batches[i] = pinnnedParticles + i * BATCH_LENGTH;
    }

    for(int i=0;i < NUM_ITERATIONS;i++){
        for (unsigned k = 0; k < size; k++) {
            Particle* selectedBatch = batches[k];
            int s_index = k % STREAM_NUM;

            cudaMemcpyAsync(&cudaParticles[k*BATCH_LENGTH], selectedBatch, batchSize, cudaMemcpyHostToDevice, streams[s_index]);
            updateParticlesKernel<<<BLOCKS, TBP, 0, streams[s_index]>>>(&cudaParticles[k*BATCH_LENGTH], TBP*BLOCKS, BATCH_LENGTH);
        }
    }
    cudaMemcpyAsync(pinnnedParticles, cudaParticles, particlesSize, cudaMemcpyDeviceToHost, streams[0]);

    for (int i = 0; i < STREAM_NUM; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }
    
    //checkConsistency(pinnnedParticles,cudaParticles,NUM_PARTICLES);

    cudaFree(cudaParticles);
    cudaFreeHost(pinnnedParticles);
    for(int i =0;i<STREAM_NUM;i++){
        cudaStreamDestroy(streams[i]);
    }
}
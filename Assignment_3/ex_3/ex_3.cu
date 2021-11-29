#include <stdio.h>
#include <cstdlib>
#include <assert.h>


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
    Particle* particles;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // Number of Particles
    unsigned NUM_PARTICLES = 100;//atoi(argv[1]);

    // Number of threads per block
    unsigned TBP = 10;//atoi(argv[2]);

    //The length of one batches
    unsigned BATCH_LENGTH = 10;//atoi(argv[3]);

    unsigned BLOCKS = (NUM_PARTICLES + TBP - 1)/TBP;

    printf("NUM_PARTICLES: %d TBP: %d BL: %d \n", NUM_PARTICLES, TBP, BATCH_LENGTH);

    size_t particlesSize = NUM_PARTICLES*sizeof(Particle);
    size_t batchSize = BATCH_LENGTH*sizeof(Particle);

    int size = (NUM_PARTICLES+BATCH_LENGTH-1)/BATCH_LENGTH;
    Particle* batches[size];
    Particle* cudaParticles;

    particles = (Particle*)malloc(particlesSize);
    cudaError_t pinnedMemory = cudaHostAlloc((void**) &cudaParticles, particlesSize, cudaHostAllocDefault);

    if (pinnedMemory != cudaSuccess) {
        printf("pinnedMemory Allocation resulted in error  %d", pinnedMemory);
    }

    generateRandomParticles(particles, NUM_PARTICLES);
//    for (int i = 0; i < 100; i++){
//        printf("%f\n",(particles+i)->position.x);
//    }

    //divide particles into batches
    for (int i = 0; i < size; i++){
        batches[i] = particles + i * BATCH_LENGTH;
//        printf("%f\n",(particles + i * BATCH_LENGTH)->position.x);
    }

    for (unsigned k = 0; k < size; k++) {

        Particle* selectedBatch = batches[k];
//        for (int j = 0; j < BATCH_LENGTH; j++){
//            printf("%f\n", selectedBatch[j].position.x);
//        }
        cudaMemcpyAsync(&cudaParticles[k*BATCH_LENGTH], selectedBatch, batchSize, cudaMemcpyHostToDevice, stream1);
        updateParticlesKernel<<<BLOCKS, TBP, 0, stream1>>>(&cudaParticles[k*BATCH_LENGTH], TBP*BLOCKS, BATCH_LENGTH);
        cudaStreamSynchronize(stream1);
        cudaMemcpyAsync(selectedBatch, &cudaParticles[k*BATCH_LENGTH], batchSize, cudaMemcpyDeviceToHost, stream1);
    }

//    updateParticles(particles,particlesSize);
//    checkConsistency(particles,cudaParticles,particlesSize);

    free(particles);
    cudaFreeHost(cudaParticles);
    cudaStreamDestroy(stream1);
}
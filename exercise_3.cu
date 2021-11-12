#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <iostream>

#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 20
#define N 64
#define TPB 32

struct Particle {
	float3 position;
	float3 velocity;
};


__global__ void updateParticlesKernel(thrust::device_vector<Particle> particles) {
	// int threadID = threadIdx.x;
	const int i = blockIdx.x  *blockDim.x + threadIdx.x;

	// (&particles[i]).get()->velocity.x += 0.1;
	// (&particles[i]).get()->velocity.y = 0.001;
	// (&particles[i]).get()->velocity.z -= 0.002; 
	// (&particles[i]).get()->position.x += (&particles[i]).get()->velocity.x * 1;


	static_cast<float> (particles[i].velocity.x) += 0.1;
	// particles[i].velocity.y = 0.001;
	// particles[i].velocity.z -= 0.002; 
	// particles[i].position.x += particles[i];

}

__host__ void updateParticles(thrust::host_vector<Particle> particles) {
	
	for (Particle particle: particles) {
		particle.velocity.x += 0.1;
		particle.velocity.y += 0.001;
		particle.velocity.z -= 0.002; 
		particle.position.x += particle.velocity.x * 1;
	}


}

__host__ thrust::host_vector<Particle> generate_random_particles(int no_particles) {
	Particle particle_zero;
	particle_zero.position = make_float3(0.0, 0.0, 0.0);
	particle_zero.velocity = make_float3(0.0, 0.0, 0.0);


	thrust::host_vector<Particle> particles (no_particles, particle_zero);
	for (unsigned int i = 0; i < no_particles; i++) {
		particles[i].position = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
		particles[i].velocity = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
	}
	return particles;
}


int main(int argc, char** argv){
	// argv[]
	thrust::host_vector<Particle> particles = generate_random_particles(100);
	updateParticles(particles);


	thrust::device_vector<Particle> *gpu_particle;
	cudaMalloc(&gpu_particle, particles.size() * sizeof(thrust::device_vector<Particle>));
	thrust::host_vector<Particle> *cpu_particle = &particles;

	cudaMemcpy(gpu_particle, cpu_particle, particles.size() * sizeof(thrust::device_vector<Particle>),cudaMemcpyHostToDevice);
	
	thrust::device_vector<Particle> particles_gpu = *gpu_particle;

	updateParticlesKernel<<<N/TPB, TPB>>>(particles_gpu);
  	cudaDeviceSynchronize();
	return 0;
}


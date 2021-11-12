#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <utility>

#include <iostream>

#define NUM_PARTICLES 10000
#define NUM_ITERATIONS 20
#define N 64
#define TPB 32

struct Particle {
	float3 position;
	float3 velocity;
};


__global__ void updateParticlesKernel(thrust::device_vector<float3>* position, thrust::device_vector<float3>* velocity) {
	// int threadID = threadIdx.x;
	const int i = blockIdx.x  *blockDim.x + threadIdx.x;

	*velocity[i].x = *velocity[i].x + 0.1;
	// *velocity[i].y += 0.001;
	// *velocity[i].z -= 0.002; 
	// *position[i].x += *velocity[i].x;


	// static_cast<float> (particles[i].velocity.x) += 0.1;
	// particles[i].velocity.y = 0.001;
	// particles[i].velocity.z -= 0.002; 
	// particles[i].position.x += particles[i];

}

__host__ void updateParticles(thrust::host_vector<float3> particles_position, thrust::host_vector<float3> particles_velocity) {
	
	for (unsigned int i = 0; i < particles_position.size(); i++) {
		particles_velocity[i].x += 0.1;
		particles_velocity[i].y += 0.001;
		particles_velocity[i].z -= 0.002; 
		particles_position[i].x += particles_velocity[i].x * 1;
	}


}

__host__ std::pair<thrust::host_vector<float3>, thrust::host_vector<float3>> generate_random_particles(int no_particles) {

	float3 particles_position_zero = make_float3(0.0, 0.0, 0.0);
	float3 particles_velocity_zero = make_float3(0.0, 0.0, 0.0);


	thrust::host_vector<float3> particles_position (no_particles, particles_position_zero);
	thrust::host_vector<float3> particles_velocity (no_particles, particles_velocity_zero);

	for (unsigned int i = 0; i < no_particles; i++) {
		particles_position[i] = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
		particles_velocity[i] = make_float3((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
	}
	return std::make_pair(particles_position, particles_velocity);
}


int main(int argc, char** argv){

	std::pair<thrust::host_vector<float3>, thrust::host_vector<float3>> particles = generate_random_particles(100);

	// updateParticles(particles.first, particles.second);


	thrust::device_vector<float3> gpuParticlePosition = particles.first;
	thrust::device_vector<float3> gpuParticleVelocity = particles.second;

	updateParticlesKernel<<<N/TPB, TPB>>>(&gpuParticlePosition, &gpuParticleVelocity);
  	// cudaDeviceSynchronize();
	return 0;
}


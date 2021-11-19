# Homework 3

# Exercise 1

# Exercise 2

## 2a. Programming exercise on Pinned Memory

1. -
2. Modified Program nvprof output
> lucasap@red-06:~/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2a 1000000 32
    NUM_PARTICLES: 1000000 TBP: 32 
    ==22330== NVPROF is profiling process 22330, command: ./ex_2a 1000000 32
    ==22330== Profiling application: ./ex_2a 1000000 32
    ==22330== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    GPU activities:   36.02%  510.69ms       200  2.5535ms  2.4558ms  3.0063ms  [CUDA memcpy DtoH]
                    35.93%  509.49ms       200  2.5475ms  2.4364ms  2.8964ms  [CUDA memcpy HtoD]
                    28.05%  397.76ms       200  1.9888ms  1.8958ms  2.6307ms  updateParticlesKernel(Particle*, unsigned int, unsigned int)
        API calls:   68.75%  1.04891s       400  2.6223ms  2.4661ms  3.1672ms  cudaMemcpy
                    27.10%  413.43ms       200  2.0672ms  1.9543ms  2.7435ms  cudaDeviceSynchronize
                        3.97%  60.602ms         1  60.602ms  60.602ms  60.602ms  cudaMalloc
                        0.15%  2.2866ms       200  11.433us  6.7160us  25.091us  cudaLaunchKernel
                        0.02%  357.14us       101  3.5360us     233ns  161.62us  cuDeviceGetAttribute
                        0.01%  131.43us         1  131.43us  131.43us  131.43us  cudaFree
                        0.00%  36.920us         1  36.920us  36.920us  36.920us  cuDeviceGetName
                        0.00%  11.114us         1  11.114us  11.114us  11.114us  cuDeviceGetPCIBusId
                        0.00%  2.2300us         3     743ns     354ns  1.4750us  cuDeviceGetCount
                        0.00%  1.3090us         2     654ns     271ns  1.0380us  cuDeviceGet    lucasap@red-06:~/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2a 1000000 32
    NUM_PARTICLES: 1000000 TBP: 32 
    ==22330== NVPROF is profiling process 22330, command: ./ex_2a 1000000 32
    ==22330== Profiling application: ./ex_2a 1000000 32
    ==22330== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    GPU activities:   36.02%  510.69ms       200  2.5535ms  2.4558ms  3.0063ms  [CUDA memcpy DtoH]
                    35.93%  509.49ms       200  2.5475ms  2.4364ms  2.8964ms  [CUDA memcpy HtoD]
                    28.05%  397.76ms       200  1.9888ms  1.8958ms  2.6307ms  updateParticlesKernel(Particle*, unsigned int, unsigned int)
        API calls:   68.75%  1.04891s       400  2.6223ms  2.4661ms  3.1672ms  cudaMemcpy
                    27.10%  413.43ms       200  2.0672ms  1.9543ms  2.7435ms  cudaDeviceSynchronize
                        3.97%  60.602ms         1  60.602ms  60.602ms  60.602ms  cudaMalloc
                        0.15%  2.2866ms       200  11.433us  6.7160us  25.091us  cudaLaunchKernel
                        0.02%  357.14us       101  3.5360us     233ns  161.62us  cuDeviceGetAttribute
                        0.01%  131.43us         1  131.43us  131.43us  131.43us  cudaFree
                        0.00%  36.920us         1  36.920us  36.920us  36.920us  cuDeviceGetName
                        0.00%  11.114us         1  11.114us  11.114us  11.114us  cuDeviceGetPCIBusId
                        0.00%  2.2300us         3     743ns     354ns  1.4750us  cuDeviceGetCount
                        0.00%  1.3090us         2     654ns     271ns  1.0380us  cuDeviceGet
                        0.00%     677ns         1     677ns     677ns     677ns  cuDeviceTotalMem
                        0.00%     452ns         1     452ns     452ns     452ns  cuDeviceGetUuid
                        0.00%     677ns         1     677ns     677ns     677ns  cuDeviceTotalMem
                        0.00%     452ns         1     452ns     452ns     452ns  cuDeviceGetUuid


2. Modified Program nvprof output



    
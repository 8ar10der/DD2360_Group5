# Homework 3

# Exercise 1

# Exercise 2

## 2a. Programming exercise on Pinned Memory

1. -
2. Modified Program nvprof output

```
lucasap@red-06:~/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2a 1000000 32
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

```

Using pinned memory... this seems wrong...
```
lucasap@red-06:~/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2a 1000000 32
NUM_PARTICLES: 1000000 TBP: 32 
==23614== NVPROF is profiling process 23614, command: ./ex_2a 1000000 32
==23614== Profiling application: ./ex_2a 1000000 32
==23614== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.74%  11.2146s       200  56.073ms  52.382ms  83.762ms  updateParticlesKernel(Particle*, unsigned int, unsigned int)
                    6.26%  748.66ms       400  1.8716ms  1.8012ms  3.0623ms  [CUDA memcpy HtoH]
      API calls:   93.11%  11.2229s       200  56.115ms  52.442ms  84.369ms  cudaDeviceSynchronize
                    6.23%  751.15ms       400  1.8779ms  1.8044ms  3.0779ms  cudaMemcpy
                    0.60%  72.604ms         1  72.604ms  72.604ms  72.604ms  cudaHostAlloc
                    0.03%  3.6778ms       200  18.388us  12.368us  35.127us  cudaLaunchKernel
                    0.02%  2.9856ms         1  2.9856ms  2.9856ms  2.9856ms  cudaFreeHost
                    0.00%  200.15us       101  1.9810us     244ns  82.814us  cuDeviceGetAttribute
                    0.00%  34.262us         1  34.262us  34.262us  34.262us  cuDeviceGetName
                    0.00%  11.889us         1  11.889us  11.889us  11.889us  cuDeviceGetPCIBusId
                    0.00%  2.3130us         3     771ns     362ns  1.2180us  cuDeviceGetCount
                    0.00%  1.4150us         2     707ns     238ns  1.1770us  cuDeviceGet
                    0.00%     548ns         1     548ns     548ns     548ns  cuDeviceTotalMem
                    0.00%     471ns         1     471ns     471ns     471ns  cuDeviceGetUuid
```


2b. Using Unified memory

```
lucasap@red-05:/afs/kth.se/home/l/u/lucasap/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2b 1000000 32
NUM_PARTICLES: 1000000 TBP: 32 
==4819== NVPROF is profiling process 4819, command: ./ex_2b 1000000 32
==4819== Profiling application: ./ex_2b 1000000 32
==4819== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  388.30ms       200  1.9415ms  1.8833ms  3.0272ms  updateParticlesKernel(Particle*, unsigned int, unsigned int)
      API calls:   73.14%  393.20ms       200  1.9660ms  1.8888ms  3.5190ms  cudaDeviceSynchronize
                   24.92%  133.98ms         1  133.98ms  133.98ms  133.98ms  cudaMallocManaged
                    1.56%  8.3656ms       200  41.827us  18.678us  2.2255ms  cudaLaunchKernel
                    0.35%  1.8687ms         1  1.8687ms  1.8687ms  1.8687ms  cudaFree
                    0.03%  164.27us       101  1.6260us     188ns  68.108us  cuDeviceGetAttribute
                    0.01%  34.954us         1  34.954us  34.954us  34.954us  cuDeviceGetName
                    0.00%  8.8800us         1  8.8800us  8.8800us  8.8800us  cuDeviceGetPCIBusId
                    0.00%  2.1770us         3     725ns     250ns  1.6210us  cuDeviceGetCount
                    0.00%  1.0600us         2     530ns     210ns     850ns  cuDeviceGet
                    0.00%     564ns         1     564ns     564ns     564ns  cuDeviceTotalMem
                    0.00%     347ns         1     347ns     347ns     347ns  cuDeviceGetUuid

==4819== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 745 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      12  1.9076MB  1.3750MB  2.0000MB  22.89063MB  2.023786ms  Host To Device
     144  162.78KB  4.0000KB  0.9961MB  22.89063MB  2.130825ms  Device To Host
Total CPU Page faults: 72
```


```
lucasap@red-05:/afs/kth.se/home/l/u/lucasap/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2_modified 10000000 32
NUM_PARTICLES: 10000000 TBP: 32 
==5991== NVPROF is profiling process 5991, command: ./ex_2_modified 10000000 32
==5991== Profiling application: ./ex_2_modified 10000000 32
==5991== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   36.76%  5.24082s       200  26.204ms  25.698ms  27.416ms  [CUDA memcpy HtoD]
                   36.75%  5.23937s       200  26.197ms  26.023ms  27.424ms  [CUDA memcpy DtoH]
                   26.49%  3.77697s       200  18.885ms  18.820ms  21.194ms  updateParticlesKernel(Particle*, unsigned int, unsigned int)
      API calls:   73.18%  10.5114s       400  26.278ms  25.788ms  27.548ms  cudaMemcpy
                   26.39%  3.79132s       200  18.957ms  18.871ms  21.284ms  cudaDeviceSynchronize
                    0.41%  58.312ms         1  58.312ms  58.312ms  58.312ms  cudaMalloc
                    0.02%  2.3142ms       200  11.571us  8.4020us  36.266us  cudaLaunchKernel
                    0.00%  379.50us         1  379.50us  379.50us  379.50us  cudaFree
                    0.00%  162.26us       101  1.6060us     156ns  68.964us  cuDeviceGetAttribute
                    0.00%  26.394us         1  26.394us  26.394us  26.394us  cuDeviceGetName
                    0.00%  8.3090us         1  8.3090us  8.3090us  8.3090us  cuDeviceGetPCIBusId
                    0.00%  1.5030us         3     501ns     248ns     921ns  cuDeviceGetCount
                    0.00%     742ns         2     371ns     166ns     576ns  cuDeviceGet
                    0.00%     367ns         1     367ns     367ns     367ns  cuDeviceTotalMem
                    0.00%     283ns         1     283ns     283ns     283ns  cuDeviceGetUuid

lucasap@red-05:/afs/kth.se/home/l/u/lucasap/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2b 10000000 32
NUM_PARTICLES: 10000000 TBP: 32 
==6029== NVPROF is profiling process 6029, command: ./ex_2b 10000000 32
==6029== Profiling application: ./ex_2b 10000000 32
==6029== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  3.77858s       200  18.893ms  18.818ms  21.179ms  updateParticlesKernel(Particle*, unsigned int, unsigned int)
      API calls:   96.33%  3.78062s       200  18.903ms  18.824ms  21.189ms  cudaDeviceSynchronize
                    2.46%  96.513ms         1  96.513ms  96.513ms  96.513ms  cudaMallocManaged
                    0.71%  27.756ms       200  138.78us  18.808us  20.502ms  cudaLaunchKernel
                    0.50%  19.682ms         1  19.682ms  19.682ms  19.682ms  cudaFree
                    0.00%  132.47us       101  1.3110us     173ns  55.171us  cuDeviceGetAttribute
                    0.00%  22.310us         1  22.310us  22.310us  22.310us  cuDeviceGetName
                    0.00%  5.9140us         1  5.9140us  5.9140us  5.9140us  cuDeviceGetPCIBusId
                    0.00%  1.7440us         3     581ns     280ns     934ns  cuDeviceGetCount
                    0.00%  1.0600us         2     530ns     192ns     868ns  cuDeviceGet
                    0.00%     442ns         1     442ns     442ns     442ns  cuDeviceTotalMem
                    0.00%     260ns         1     260ns     260ns     260ns  cuDeviceGetUuid

==6029== Unified Memory profiling result:
Device "NVIDIA GeForce GTX 745 (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     115  1.9903MB  1.3750MB  2.0000MB  228.8828MB  20.13415ms  Host To Device
    1380  169.84KB  4.0000KB  0.9961MB  228.8828MB  21.28750ms  Device To Host
Total CPU Page faults: 690



lucasap@red-05:/afs/kth.se/home/l/u/lucasap/repos/DD2360_Group5/Assignment_3/ex_2$ nvprof ./ex_2a 10000000 32
NUM_PARTICLES: 10000000 TBP: 32 
==5248== NVPROF is profiling process 5248, command: ./ex_2a 10000000 32
==5248== Profiling application: ./ex_2a 10000000 32
==5248== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   93.24%  114.819s       200  574.09ms  529.17ms  714.87ms  updateParticlesKernel(Particle*, unsigned int, unsigned int)
                    6.76%  8.32889s       400  20.822ms  19.269ms  31.366ms  [CUDA memcpy HtoH]
      API calls:   93.14%  114.829s       200  574.14ms  529.18ms  714.94ms  cudaDeviceSynchronize
                    6.76%  8.33351s       400  20.834ms  19.288ms  31.381ms  cudaMemcpy
                    0.08%  99.974ms         1  99.974ms  99.974ms  99.974ms  cudaHostAlloc
                    0.02%  22.939ms         1  22.939ms  22.939ms  22.939ms  cudaFreeHost
                    0.00%  5.4104ms       200  27.052us  20.748us  71.272us  cudaLaunchKernel
                    0.00%  133.03us       101  1.3170us     150ns  55.657us  cuDeviceGetAttribute
                    0.00%  24.319us         1  24.319us  24.319us  24.319us  cuDeviceGetName
                    0.00%  5.6360us         1  5.6360us  5.6360us  5.6360us  cuDeviceGetPCIBusId
                    0.00%  1.3210us         3     440ns     207ns     862ns  cuDeviceGetCount
                    0.00%     937ns         2     468ns     150ns     787ns  cuDeviceGet
                    0.00%     493ns         1     493ns     493ns     493ns  cuDeviceTotalMem
                    0.00%     254ns         1     254ns     254ns     254ns  cuDeviceGetUuid

```


# Exercise 3

# Exercise Bonus


```
lucasap@red-14:/afs/kth.se/home/l/u/lucasap/repos/DD2360_Group5/Assignment_3/ex_bonus$ ./exercise_3.out ./exercise_3.out -s 1024 -v
Matrix size: 1024x1024
Matrix size: 1024x1024
Grid size: 64x64
Tile size: 16x16
Run CPU sgemm: 1

CPU matmul:			158.873000 ms
GPU cuBLAS matmul:		3.689000 ms
GPU matmul (global memory):	219.479000 ms
GPU matmul (shared memory):	22.277000 ms

```
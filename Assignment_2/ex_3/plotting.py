import matplotlib.pyplot as plt
import numpy as np

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Palatino"],
# })

def plot_exercise_3_cpu():
    num_particles = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    cpu_times = [0.014892, 0.074124, 0.146147, 0.769355, 1.593504, 7.787810, 15.730628]
    plt.plot(num_particles, cpu_times)
    # plt.xscale("log")
    plt.xlabel("Numbers of Particles")
    plt.xlabel("Execution time [s]")
    plt.show()

def plot_exercise_3_comparision():
    num_particles = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    cpu_times = [0.014949, 0.072594, 0.142762, 0.760029, 1.527007, 7.668389, 15.296968]
    gpu_times = [0.004744, 0.024513, 0.046592, 0.191766, 0.381050, 1.903732, 3.807285]

    cpu_line = plt.plot(num_particles, cpu_times, label="CPU", marker='x')
    gpu_line = plt.plot(num_particles, gpu_times, label="GPU", marker='x')
    plt.grid(True)
    plt.title(f"Execution time against Number of Particles for the CPU and GPU implementation", fontsize=18)

    plt.xlabel("Numbers of Particles", fontsize=14)
    plt.ylabel("Execution time [s]", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(["CPU", "GPU"])
    plt.show()

def plot_exercise_3_gpu():
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')


    # Data for a three-dimensional line
    BLOCK_SIZE = np.array([16, 32, 64, 128, 256])
    NUM_PARTICLES = np.array([10000, 100000, 1000000, 10000000])
    zdata = np.array([
                [0.003855, 0.004563, 0.004028, 0.003961, 0.003983],
                [0.038237, 0.038011, 0.038255, 0.038780, 0.038715],
                [0.377732, 0.378038, 0.379140, 0.378877, 0.432408],
                [4.041089, 3.779580, 3.979433, 3.806148, 3.806210]
            ])
    for i, y in enumerate(NUM_PARTICLES):
        plt.plot(BLOCK_SIZE, zdata[i, :], label=BLOCK_SIZE, marker='x')
        plt.title(f"Different Execution times for NUM_PARTICLES={y}\n for different block sizes")
        plt.tight_layout()
        plt.grid(True)
        plt.xlabel("Block Size")
        plt.ylabel("Execution Time [s]")
        plt.savefig(f"ex3_block_sizes_particles_{y}", dpi=300, bbox_inches='tight')
        plt.clf()
    # ax.scatter3D(NUM_PARTICLES, BLOCK_SIZE, zdata, c=zdata, cmap='Greens')


if __name__ == "__main__":
    # plot_exercise_3_comparison()
    plot_exercise_3_gpu()

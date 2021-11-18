import matplotlib.pyplot as plt

def plot_exercise_3_cpu():
    num_particles = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    cpu_times = [0.014892, 0.074124, 0.146147, 0.769355, 1.593504, 7.787810, 15.730628]
    plt.plot(num_particles, cpu_times)
    # plt.xscale("log")
    plt.show()

def plot_exercise_3_gpu():
    pass

def plot_exercise_3_comparison():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    xdata = [16, 32, 64, 128, 256]
    ydata = [10000, 100000, 1000000, 10000000]
    zdata = [
            0.003855, 0.004563, 0.004028, 0.003961, 0.003983,
            0.038237, 0.038011, 0.038255, 0.038780, 0.038715,
            0.377732, 0.378038, 0.379140, 0.378877, 0.432408,
            4.041089, 3.779580, 3.979433, 3.806148, 3.806210,
            ]

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')


if __name__ == "__main__":
    plot_exercise_3_comparison()
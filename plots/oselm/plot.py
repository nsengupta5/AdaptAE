from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

def plot_time_vs_prop(data):
    # Creating two separate plots for Initial Time and Sequential Time
    _, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for Initial Time
    ax[0].plot(data['Sequential Prop'], data['Initial Time'], label='Initial Time', marker='o', color='blue')
    ax[0].set_title('Sequential Prop vs Initial Time')
    ax[0].set_xlabel('Sequential Prop')
    ax[0].set_ylabel('Initial Time')
    ax[0].grid(True)

    # Plot for Sequential Time
    ax[1].plot(data['Sequential Prop'], data['Sequential Time'], label='Sequential Time', marker='x', color='green')
    ax[1].set_title('Sequential Prop vs Sequential Time')
    ax[1].set_xlabel('Sequential Prop')
    ax[1].set_ylabel('Sequential Time')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/oselm/sequential_prop_vs_time.png')

def plot_peak_memory_vs_prop(data):
    _, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for Initial Peak Memory
    ax[0].plot(data['Sequential Prop'], data['Initial Peak Memory'], label='Initial Peak Memory', marker='o', color='blue')
    ax[0].set_title('Sequential Prop vs Initial Peak Memory')
    ax[0].set_xlabel('Sequential Prop')
    ax[0].set_ylabel('Initial Peak Memory')
    ax[0].grid(True)

    # Plot for Sequential Peak Memory
    ax[1].plot(data['Sequential Prop'], data['Sequential Peak Memory'], label='Sequential Peak Memory', marker='x', color='green')
    ax[1].set_title('Sequential Prop vs Sequential Peak Memory')
    ax[1].set_xlabel('Sequential Prop')
    ax[1].set_ylabel('Sequential Peak Memory')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/oselm/sequential_prop_vs_peak_memory.png')

def plot_loss_vs_prop(data):
    # Creating two separate plots for Initial Loss and Sequential Loss
    _, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for Initial Loss
    ax[0].plot(data['Sequential Prop'], data['Initial Loss'], label='Initial Loss', marker='o', color='blue')
    ax[0].set_title('Sequential Prop vs Initial Loss')
    ax[0].set_xlabel('Sequential Prop')
    ax[0].set_ylabel('Initial Loss')
    ax[0].grid(True)

    # Plot for Sequential Loss
    ax[1].plot(data['Sequential Prop'], data['Sequential Loss'], label='Sequential Loss', marker='x', color='green')
    ax[1].set_title('Sequential Prop vs Sequential Loss')
    ax[1].set_xlabel('Sequential Prop')
    ax[1].set_ylabel('Sequential Loss')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/oselm/sequential_prop_vs_loss.png')

def plot_time_vs_batch(data):
    # Creating two separate plots for Initial Time and Sequential Time
    _, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for Initial Time
    ax[0].plot(data['Batch Size'], data['Initial Time'], label='Initial Time', marker='o', color='blue')
    ax[0].set_title('Batch Size vs Initial Time')
    ax[0].set_xlabel('Batch Size')
    ax[0].set_ylabel('Initial Time')
    ax[0].grid(True)

    # Plot for Sequential Time
    ax[1].plot(data['Batch Size'], data['Sequential Time'], label='Sequential Time', marker='x', color='green')
    ax[1].set_title('Batch Size vs Sequential Time')
    ax[1].set_xlabel('Batch Size')
    ax[1].set_ylabel('Sequential Time')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/oselm/batch_size_vs_time.png')

def plot_peak_memory_vs_batch(data):
    _, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for Initial Peak Memory
    ax[0].plot(data['Batch Size'], data['Initial Peak Memory'], label='Initial Peak Memory', marker='o', color='blue')
    ax[0].set_title('Batch Size vs Initial Peak Memory')
    ax[0].set_xlabel('Batch Size')
    ax[0].set_ylabel('Initial Peak Memory')
    ax[0].grid(True)

    # Plot for Sequential Peak Memory
    ax[1].plot(data['Batch Size'], data['Sequential Peak Memory'], label='Sequential Peak Memory', marker='x', color='green')
    ax[1].set_title('Batch Size vs Sequential Peak Memory')
    ax[1].set_xlabel('Batch Size')
    ax[1].set_ylabel('Sequential Peak Memory')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/oselm/batch_size_vs_peak_memory.png')

def plot_loss_vs_batch(data):
    _, ax = plt.subplots(2, 1, figsize=(10, 12))

    # Plot for Initial Loss
    ax[0].plot(data['Batch Size'], data['Initial Loss'], label='Initial Loss', marker='o', color='blue')
    ax[0].set_title('Batch Size vs Initial Loss')
    ax[0].set_xlabel('Batch Size')
    ax[0].set_ylabel('Initial Loss')
    ax[0].grid(True)

    # Plot for Sequential Loss
    ax[1].plot(data['Batch Size'], data['Sequential Loss'], label='Sequential Loss', marker='x', color='green')
    ax[1].set_title('Batch Size vs Sequential Loss')
    ax[1].set_xlabel('Batch Size')
    ax[1].set_ylabel('Sequential Loss')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig('plots/oselm/batch_size_vs_loss.png')
    pass


def plot_total_all_vs_peak_mem(data):
    # Creating a 3D plot for Sequential Prop, Batch Size, and Peak Memory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = data['Sequential Prop']
    y = data['Batch Size']
    z = data['Total Peak Memory']

    ax.scatter(x, y, z, c='green', marker='o')

    ax.set_xlabel('Sequential Prop')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Total Peak Memory (MB)')

    plt.title('3D Plot: Sequential Prop, Batch Size, and Peak Memory')
    plt.show()

def plot_total_all_vs_time(data):
    # Creating a 3D plot for Sequential Prop, Batch Size, and Peak Memory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = data['Sequential Prop']
    y = data['Batch Size']
    z = data['Total Time']

    ax.scatter(x, y, z, c='green', marker='o')

    ax.set_xlabel('Sequential Prop')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Total Time (s)')

    plt.title('3D Plot: Sequential Prop, Batch Size, and Training Time')
    plt.show()

def plot_total_all_vs_loss(data):
    # Creating a 3D plot for Sequential Prop, Batch Size, and Peak Memory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = data['Sequential Prop']
    y = data['Batch Size']
    z = data['Total Loss']

    ax.scatter(x, y, z, c='green', marker='o')

    ax.set_xlabel('Sequential Prop')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Total Loss')

    plt.title('3D Plot: Sequential Prop, Batch Size, and Loss')
    plt.show()

def main():
    # # Reading data from csv file
    # data = pd.read_csv('oselm/data/seq_prop_fashion-mnist_performance.csv')

    # # Plotting the data
    # plot_time_vs_prop(data)
    # plot_peak_memory_vs_prop(data)
    # plot_loss_vs_prop(data)

    # data = pd.read_csv('oselm/data/batch_size_fashion-mnist_performance.csv')

    # plot_time_vs_batch(data)
    # plot_peak_memory_vs_batch(data)
    # plot_loss_vs_batch(data)

    data = pd.read_csv('oselm/data/total/total_cifar100_performance.csv')

    plot_total_all_vs_peak_mem(data)
    plot_total_all_vs_time(data)
    plot_total_all_vs_loss(data)

if __name__ == '__main__':
    main()

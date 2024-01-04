import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_total_batch_vs_memory_sample(data):
    data = data.groupby('Batch Size').mean().reset_index()
    plt.figure(figsize=(10, 6))
    min_peak_mem_index = data['Total Peak Memory'].idxmin()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.axvline(data['Batch Size'][min_peak_mem_index], color='red', linestyle='--', label="Minimum Peak Memory")
    plt.text(
        data['Batch Size'][min_peak_mem_index] + 2.9, data['Total Peak Memory'][min_peak_mem_index] + 2.4, 'Batch Size: ' + str(data['Batch Size'][min_peak_mem_index])
    )
    sns.scatterplot(x='Batch Size', y='Total Peak Memory', data=data)
    plt.xlabel('Batch Size')
    plt.ylabel('Peak Memory Usage (MB)')
    plt.legend()
    plt.title("Super Tiny ImageNet")
    plt.show()

def plot_total_batch_vs_memory_batch(data):
    _, axs = plt.subplots(2, 3, figsize=(10, 6))

    conditions = [(data['Sequential Prop'] < 0.45),
                  (data['Sequential Prop'] == 0.55),
                  (data['Sequential Prop'] > 0.45)]
    titles = [f"(Sequential Prop < 0.5)",
              f"(Sequential Prop == 0.5)",
              f"(Sequential Prop > 0.5)"]

    for ax, condition, title in zip(axs, conditions, titles):
        # Apply the condition to filter the data
        subset_data = data[data['Batch Size'] != 1]
        subset_data = subset_data[condition]
        subset_data = subset_data.groupby('Batch Size').mean().reset_index()
        min_peak_mem_index = subset_data['Total Peak Memory'].idxmin()

        # Plotting the filtered data
        sns.scatterplot(ax=ax, x='Batch Size', y='Total Peak Memory', data=subset_data)

        # Adding grid, vertical line, text, and labels
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.axvline(subset_data['Batch Size'][min_peak_mem_index], color='red', linestyle='--', label="Minimum Peak Memory")
        ax.text(
            subset_data['Batch Size'][min_peak_mem_index] + 0.1, subset_data['Total Peak Memory'][min_peak_mem_index] + 0.1, 'Batch Size: ' + str(subset_data['Batch Size'][min_peak_mem_index])
        )
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Peak Memory Usage (MB)')
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_total_batch_vs_loss(datasets, names):
    _, axs = plt.subplots(2, 3, figsize=(15, 5))

    for i, (ax, data) in enumerate(zip(axs.flatten(), datasets)):
        data = data.groupby('Batch Size').mean().reset_index()
        sns.scatterplot(ax=ax, x='Batch Size', y='Test Loss', data=data)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Loss')

        # Set the caption using the `set_title` method or `text` method
        ax.set_title(names[i], fontsize=10)  # This will be the title

    plt.tight_layout()
    plt.show()

def plot_total_seq_prop_vs_loss(datasets, names):
    _, axs = plt.subplots(2, 3, figsize=(15, 5))

    for i, (ax, data) in enumerate(zip(axs.flatten(), datasets)):
        data = data.groupby('Sequential Prop').mean().reset_index()
        sns.scatterplot(ax=ax, x='Sequential Prop', y='Test Loss', data=data)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xlabel('Sequential Proportion')
        ax.set_ylabel('Loss')

        # Set the caption using the `set_title` method or `text` method
        ax.set_title(names[i], fontsize=10)  # This will be the title

    plt.tight_layout()
    plt.show()

def plot_total_seq_prop_vs_memory_batch(datasets, names):
    _, axs = plt.subplots(2, 3, figsize=(20, 6))

    for i, (ax, data) in enumerate(zip(axs.flatten(), datasets)):
        data = data[data['Batch Size'] != 1]
        data = data.groupby(['Sequential Prop']).mean().reset_index()
        min_peak_mem_index = data['Total Peak Memory'].idxmin()
        sns.scatterplot(ax=ax, x='Sequential Prop', y='Total Peak Memory', data=data)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.axvline(data['Sequential Prop'][min_peak_mem_index], color='red', linestyle='--', label="Minimum Peak Memory")
        ax.text(data['Sequential Prop'][min_peak_mem_index] + 0.01, data['Total Peak Memory'][min_peak_mem_index] + 0.01, 'Sequential Prop: ' + str(data['Sequential Prop'][min_peak_mem_index]))
        ax.set_xlabel('Sequential Proportion')
        ax.set_ylabel('Peak Memory Usage (MB)')
        ax.legend()
        
        # Set the caption using the `set_title` method or `text` method
        ax.set_title(names[i], fontsize=10)  # This will be the title

    plt.tight_layout()
    plt.show()

def plot_total_seq_prop_vs_memory_sample(datasets, names):
    _, axs = plt.subplots(2, 3, figsize=(15, 5))

    for i, (ax, data) in enumerate(zip(axs.flatten(), datasets)):
        data = data[data['Batch Size'] == 1]

        data = data.groupby(['Sequential Prop']).mean().reset_index()

        memory_data = data['Total Peak Memory']
        seq_data = data['Sequential Prop']
        m, b = np.polyfit(seq_data, memory_data, 1)  # m = slope, b = intercept
        ax.plot(seq_data, m*seq_data + b, color='lightblue', linestyle='-', linewidth=2, alpha=0.5)  # Plot the regression line

        sns.scatterplot(ax=ax, x='Sequential Prop', y='Total Peak Memory', data=data)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xlabel('Sequential Proportion')
        ax.set_ylabel('Peak Memory Usage (MB)')

        # Set the caption using the `set_title` method or `text` method
        ax.set_title(names[i], fontsize=10)  # This will be the title

        equation_text = f'y = {m:.0f}x + {b:.0f}'
        ax.text(0.65, 0.9, equation_text, transform=ax.transAxes, fontsize=10, color='black')

    plt.tight_layout()
    plt.show()


def plot_total_batch_vs_time(datasets, names):
    _, axs = plt.subplots(2, 3, figsize=(20, 6))

    for i, (ax, data) in enumerate(zip(axs.flatten(), datasets)):
        data = data.groupby(['Batch Size']).mean().reset_index()
        sns.scatterplot(ax=ax, x='Batch Size', y='Total Training Time', data=data)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Total Training Time (s)')
        ax.set_title(names[i], fontsize=10)  # This will be the title

    plt.tight_layout()
    plt.show()

def plot_total_seq_prop_vs_time(datasets, names):
    _, axs = plt.subplots(2, 3, figsize=(20, 6))

    for i, (ax, data) in enumerate(zip(axs.flatten(), datasets)):
        batch_data = data[data['Batch Size'] != 1]
        sample_data = data[data['Batch Size'] == 1]
        batch_data = batch_data.groupby(['Sequential Prop']).mean().reset_index()
        sample_data = sample_data.groupby(['Sequential Prop']).mean().reset_index()

        sample_seq_data = sample_data['Sequential Prop']
        sample_time_data = sample_data['Total Training Time']
        batch_seq_data = batch_data['Sequential Prop']
        batch_time_data = batch_data['Total Training Time']

        sns.scatterplot(ax=ax, x='Sequential Prop', y='Total Training Time', data=batch_data, label='Batch Size > 1')
        sns.scatterplot(ax=ax, x='Sequential Prop', y='Total Training Time', data=sample_data, label='Batch Size = 1')

        m_s, b_s = np.polyfit(sample_seq_data, sample_time_data, 1)  # m = slope, b = intercept
        ax.plot(sample_seq_data, m_s*sample_seq_data + b_s, color='orange', linestyle='-', linewidth=2, alpha=0.5)  # Plot the regression line

        m_b, b_b = np.polyfit(batch_seq_data, batch_time_data, 1)  # m = slope, b = intercept
        ax.plot(batch_seq_data, m_b*batch_seq_data + b_b, color='lightblue', linestyle='-', linewidth=2, alpha=0.5)  # Plot the regression line

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
        ax.set_xlabel('Sequential Proportion')
        ax.set_ylabel('Total Training Time (s)')
        ax.legend()
        ax.set_title(names[i], fontsize=10)  # This will be the title

        sample_equation_text = f'y = {m_s:.0f}x + {b_s:.0f}'
        ax.text(0.65, 0.9, sample_equation_text, transform=ax.transAxes, fontsize=10, color='black')
        batch_equation_text = f'y = {m_b:.0f}x + {b_b:.0f}'
        ax.text(0.65, 0.2, batch_equation_text, transform=ax.transAxes, fontsize=10, color='black')


    plt.tight_layout()
    plt.show()

def main():
    mnist_data = pd.read_csv('adaptae/data/total/total_mnist_reconstruction_performance.csv')
    fashion_mnist_data = pd.read_csv('adaptae/data/total/total_fashion-mnist_reconstruction_performance.csv')
    cifar10_data = pd.read_csv('adaptae/data/total/total_cifar10_reconstruction_performance.csv')
    cifar100_data = pd.read_csv('adaptae/data/total/total_cifar100_reconstruction_performance.csv')
    super_tiny_imagenet_data = pd.read_csv('adaptae/data/total/total_super-tiny-imagenet_reconstruction_performance.csv')
    tiny_imagenet_data = pd.read_csv('adaptae/data/total/total_tiny-imagenet_reconstruction_performance.csv')

    datasets = [mnist_data, fashion_mnist_data, cifar10_data, cifar100_data, super_tiny_imagenet_data, tiny_imagenet_data] 
    dataset_names = ['MNIST', 'Fashion MNIST', 'CIFAR-10', 'CIFAR-100', 'Super Tiny ImageNet', 'Tiny ImageNet']

    # plot_total_seq_prop_vs_memory_batch(datasets, dataset_names)
    # plot_total_seq_prop_vs_memory_sample(datasets, dataset_names)
    # plot_total_batch_vs_memory_batch(mnist_data)
    # plot_total_batch_vs_memory_sample(super_tiny_imagenet_data)
    # plot_total_batch_vs_time(datasets, dataset_names)
    # plot_total_seq_prop_vs_time(datasets, dataset_names)
    plot_total_batch_vs_loss(datasets, dataset_names)
    # plot_total_seq_prop_vs_loss(datasets, dataset_names)

if __name__ == '__main__':
    main()

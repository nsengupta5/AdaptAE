import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ae_data_dir = "autoencoder/data/"
elm_ae_data_dir = "elmae/data/"
adaptae_data_dir = "pselmae/data/total/"

model_dirs = {
    "AdaptAE": adaptae_data_dir,
    "Autoencoder": ae_data_dir,
    "ELM-AE": elm_ae_data_dir
}

colors = ['red', 'blue', 'purple']

# Paths to your CSV files
datasets = {
    "MNIST": "total_mnist_reconstruction_performance.csv",
    "Fashion-MNIST": "total_fashion-mnist_reconstruction_performance.csv",
    "CIFAR10": "total_cifar10_reconstruction_performance.csv",
    "CIFAR100": "total_cifar100_reconstruction_performance.csv",
    "Super Tiny-ImageNet": "total_super-tiny-imagenet_reconstruction_performance.csv",
    "Tiny-ImageNet": "total_tiny-imagenet_reconstruction_performance.csv"
}

bar_width = 0.3
n_groups = len(datasets)
dataset_spacing = 0.1  # Space between groups of bars
group_width = len(model_dirs) * bar_width + dataset_spacing
index = np.arange(0, n_groups * group_width, group_width)

# Dictionary to hold memory usage data
memory_usage = {model: [] for model in model_dirs}

# Read each CSV and extract memory usage
for dataset_name, dataset_path in datasets.items():
    for model, model_dir in model_dirs.items():
        df = pd.read_csv(model_dir + dataset_path)

        if model == "AdaptAE":
            df = df[df['Batch Size'] == 1]  
            df = df[df['Sequential Prop'] == 0.97]

        average_memory_usage = df['Total Training Time'].mean()
        memory_usage[model].append(df['Total Training Time'].mean())

# Plotting the bar chart
plt.figure(figsize=(12, 6))

for i, model in enumerate(model_dirs):
    bars = plt.bar(index + i * bar_width, memory_usage[model], bar_width, label=model, color=colors[i])
     # If ELM-AE could not run on Tiny ImageNet, apply a pattern to that bar.
    if model == "ELM-AE":
        # Apply a pattern like a cross hatch to indicate it didn't run.
        bars[-1].set_hatch('x')  # The last bar in the ELM-AE group corresponds to Tiny ImageNet
        bars[-1].set_edgecolor('red')  # Optional: Set the edge color to make it stand out.

    for bar in bars:
        height = bar.get_height()
        if height > 0:  # If the bar has height greater than 0, annotate it.
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                     ha='center', va='bottom')

plt.ylabel('Average Training Time (s)')
plt.xticks(index + bar_width, datasets.keys(), rotation=45)
# Add grid lines for better readability, only for the y-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

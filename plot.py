import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ae_data_dir = "autoencoder/data/"
elm_ae_data_dir = "elmae/data/"
adaptae_data_dir = "pselmae/data/total/"

models = ["Autoencoder", "ELM-AE", "AdaptAE"]

model_dirs = {
    "Autoencoder": ae_data_dir,
    "ELM-AE": elm_ae_data_dir,
    "AdaptAE": adaptae_data_dir
}

# Paths to your CSV files
datasets = {
    "MNIST": "total_mnist_reconstruction_performance.csv",
    "CIFAR100": "total_cifar100_reconstruction_performance.csv",
    "Super Tiny-ImageNet": "total_super-tiny-imagenet_reconstruction_performance.csv",
    "Tiny-ImageNet": "total_tiny-imagenet_reconstruction_performance.csv"
}

# Data for plotting
n_groups = len(datasets)
index = np.arange(n_groups)
bar_width = 0.2

# Dictionary to hold memory usage data
memory_usage = {model: [] for model in model_dirs}

# Read each CSV and extract memory usage
for dataset_name, dataset_path in datasets.items():
    for model, model_dir in model_dirs.items():
        df = pd.read_csv(model_dir + dataset_path)

        if model == "AdaptAE":
            df = df[df['Batch Size'] == 1]  
            df = df[df['Sequential Prop'] == 0.97]

        average_memory_usage = df['Total Peak Memory'].mean()
        memory_usage[model].append(df['Total Peak Memory'].mean())

# Plotting the bar chart
plt.figure(figsize=(12, 6))

for i, model in enumerate(model_dirs):
    plt.bar(index + i * bar_width, memory_usage[model], bar_width, label=model)

plt.xlabel('Datasets')
plt.ylabel('Average Memory Usage')
plt.title('Average Memory Usage by Model and Dataset')
plt.xticks(index + bar_width, datasets.keys(), rotation=45)
plt.legend()

plt.tight_layout()
plt.show()

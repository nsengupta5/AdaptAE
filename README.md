## AdaptAE: Adaptive Auteoncoder Training Framework for Edge Devices with Variable Memory Constraints

### Introduction
AdaptAE is a framework that can be employed for real-time training of autoencoders on edge devices with limited memory capacities. It is designed to be flexible, allowing ML practitioners to analytically determine the best parameters based on the resources constraints of the desired edge device whilst guaranteeing a significantly faster training time compared to that of a traditional autoencoder. 

AdaptAE was developed at the University of St Andrews as part of a Master's thesis project in the Computer Science department.

### Code Structure
The repository contains the source code of AdaptAE. The code is organised as:

The AdaptAE training code can be found in the `adaptae` folder
The autoencoder training code can be found in the `autoencoder` folder
The ELM-AE training code can be found in the `elmae` folder

The supported datasets are as follows:
- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- Tiny ImageNet

### Setting up the environment

#### Docker Environment

The simplest way to run the code is by using the pre-configured Docker container which encapsulates all the necessary dependencies and environment settings.

##### Prerequisites

Before you begin, make sure that you have Docker installed on your system. If you want to enable the Docker to utilize the NVIDIA GPU, you will also need to download the NVIDIA Container Toolkit

1. Clone the repository:

```bash
git clone https://github.com/nsengupta5/AdaptAE.git
cd AdaptAE
```

2. Build the Docker image:

```bash
docker build -t AdaptAE .
```

3. Run the Docker container:

```bash
docker run -it AdaptAE
```

If you have a NVIDIA GPU, run:

```bash
docker run --gpus all -it AdaptAE
```

4. Activate the AdaptAE conda environment:

```bash
conda activate adaptae
```

#### Manual Setup

If you would rather not use Docker, you can configure your local environment to run the AdaptAE project

##### Prerequisites

Before you begin, make sure that you have either Conda or Miniconda installed on your system

1. Clone the repository:

```bash
git clone https://github.com/nsengupta5/AdaptAE.git
cd AdaptAE
```

2. Create the Conda environment:

```bash
conda env create -f environment.yml -n adaptae
```

3. Activate the environment:
```bash
conda activate adaptae
```

4. Set the PYTHONPATH variable:

```bash
export PYTHONPATH="./:$PYTHONPATH"
```

### Running the code
Running the AdaptAE model: Please follow the instructions in the `adaptae` folder
Running the autoencoder model: Please follow the instructions in the `autoencoder` folder
Running the ELM-AE model: Please follow the instructions in the `elmae` folder

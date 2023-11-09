import torch
from oselm import OSELM, assert_cond
from torchvision import datasets, transforms
from torch.backends import mps
from torch import cuda, clamp, set_printoptions
from torch.utils.data import random_split
from sys import argv
import logging
import time

# Constants
TRAIN_SIZE_PROP = 0.6
SEQ_SIZE_PROP = 0.2
TEST_SIZE_PROP = 0.2
BATCH_SIZE = 64
CACHE_BUFFER = 3000
DEVICE = (
    "cuda"
    if cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

"""
Initialize the OSELM model
"""
def oselm_init(input_nodes, hidden_nodes):
    logging.info(f"Initializing OSELM model...")
    activation_func = 'sigmoid'
    loss_func = 'mse'
    logging.info(f"Initializing OSELM model complete.")
    return OSELM(activation_func, loss_func, input_nodes, hidden_nodes, DEVICE).to(DEVICE)

"""
Load and split the data into training, sequential and test data
"""
def load_and_split_data(dataset):
    logging.info(f"Loading and preparing data...")
    transform = transforms.ToTensor()
    input_nodes = 784
    hidden_nodes = 128
    match dataset:
        case 'mnist':
            transform = transforms.ToTensor()
            data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
        case 'fashion-mnist':
            transform = transforms.ToTensor()
            data = datasets.FashionMNIST(root = './data', train = True, download = True, transform = transform)
        case 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            data = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            data = datasets.CIFAR100(root = './data', train = True, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    # Split 60% for training, 20% for sequential training and 20% for testing
    train_size = int(TRAIN_SIZE_PROP * len(data))
    seq_size = int(SEQ_SIZE_PROP * len(data))
    test_size = len(data) - train_size - seq_size
    train_data, seq_data, test_data = random_split(data, [train_size, seq_size, test_size])
    logging.info(f"Loading and preparing data complete.")
    return train_data, seq_data, test_data, input_nodes, hidden_nodes

"""
Initialize the OSELM model with the initial training data
:param model: The OSELM model
:param train_data: The initial training data
"""
def train_init(model, train_data):
    # Assert that the initial training data is of the correct shape
    data = torch.tensor(train_data.dataset.data).float().to(DEVICE) / 255
    data = data.reshape(-1, model.input_shape[0])
    assert_cond(data.shape[0] == len(train_data.dataset), "Train data shape mismatch")
    logging.info(f"Initial training on {len(data)} samples...")
    logging.info("Train data shape: " + str(data.shape))

    # Time and CUDA memory tracking
    start_time = time.time()
    initial_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    model.init_phase(data)
    end_time = time.time()
    training_time = end_time - start_time
    
    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    print("\nInitial Training Benchmarks:")
    print("===========================================")
    print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
    print(f"Memory used during training: {(final_memory - initial_memory) / (1024 ** 2):.2f} MB")
    print(f"Initial Training complete. Time taken: {training_time:.2f} seconds.\n")

    logging.info(f"Initial training complete")

"""
Train the OSELM model sequentially on the sequential training data
:param model: The OSELM model
:param seq_data: The sequential training data
:param mode: The mode of sequential training, either "sample" or "batch"
"""
def train_sequential(model, seq_data, mode):
    logging.info(f"Sequential training on {len(seq_data.dataset)} samples in {mode} mode...")
    data = torch.tensor(seq_data.dataset.data).float().to(DEVICE) / 255
    data = data.reshape(-1, model.input_shape[0])
    assert_cond(data.shape[0] == len(seq_data.dataset), "Sequential data shape mismatch")
    logging.info("Sequential data shape: " + str(data.shape))
    if mode == "sample":
        # Time and CUDA memory tracking
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()

        for idx, image in enumerate(data):
            model.seq_phase(image, mode)
            if idx % CACHE_BUFFER == 0:
                cuda.empty_cache()
        end_time = time.time()
        training_time = end_time - start_time
        
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print("\nSequential Training Benchmarks:")
        print("===========================================")
        print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
        print(f"Memory used during training: {(final_memory - initial_memory) / (1024 ** 2):.2f} MB")
        print(f"Sequential training complete. Time taken: {training_time:.2f} seconds.\n")
    else:
        # Time and CUDA memory tracking
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()

        for i in range(0, len(data), BATCH_SIZE):
            images = data[i:i+BATCH_SIZE]
            model.seq_phase(images, mode)

        end_time = time.time()
        training_time = end_time - start_time
        
        final_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print("\nSequential Training Benchmarks:")
        print("===========================================")
        print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
        print(f"Memory used during training: {(final_memory - initial_memory) / (1024 ** 2):.2f} MB")
        print(f"Sequential training complete. Time taken: {training_time:.2f} seconds.\n")

    logging.info(f"Sequential training complete")

"""
Test the OSELM model on the test data
:param model: The OSELM model
:param test_data: The test data
"""
def test_model(model, test_data):
    logging.info(f"Testing on {len(test_data.dataset)} samples...")
    set_printoptions(sci_mode=False)
    data = torch.tensor(test_data.dataset.data).float().to(DEVICE) / 255
    data = data.reshape(-1, model.input_shape[0])
    assert_cond(data.shape[0] == len(test_data.dataset), "Test data shape mismatch")
    pred = model.predict(data)
    pred = clamp(pred, min=0).round().int()
    loss, _ = model.evaluate(data, pred)
    print(f"Loss: {loss.item():.5f}")
    logging.info(f"Testing complete.")

"""
Exit the program with an error message of the correct usage
"""
def exit_with_error():
    print("Usage: python train-os-elm.py [mode] [dataset]")
    print("mode: sample or batch")
    print("dataset: mnist, fashion-mnist, cifar10 or cifar100")
    exit(1)

"""
Get the mode of sequential training (either "sample" or "batch")
"""
def get_mode():
    if len(argv) == 1:
        # Default to sample mode
        return "sample"
    elif argv[1] not in ["sample", "batch"]:
        exit_with_error()
    else:
        return argv[1]

"""
Get the dataset to use (either "mnist", "fashion-mnist", "cifar10" or "cifar100")
"""
def get_dataset():
    if len(argv) < 3:
        # Default to MNIST datasets
        return "mnist"
    elif argv[2] not in ["mnist", "fashion-mnist", "cifar10", "cifar100"]:
        exit_with_error()
    else:
        return argv[2]

def main():
    mode = get_mode()
    dataset = get_dataset()
    logging.basicConfig(level=logging.INFO)
    train_data, seq_data, test_data, input_nodes, hidden_nodes = load_and_split_data(dataset)
    model = oselm_init(input_nodes, hidden_nodes)

    # Time and CUDA memory tracking
    start_time = time.time()
    initial_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()

    train_init(model, train_data)
    train_sequential(model, seq_data, mode)

    end_time = time.time()
    training_time = end_time - start_time
    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    print("\nTotal Training Benchmarks:")
    print("===========================================")
    print(f"Peak memory allocated during total training: {peak_memory / (1024 ** 2):.2f} MB")
    print(f"Memory used during total training: {(final_memory - initial_memory) / (1024 ** 2):.2f} MB")
    print(f"Total training complete. Time taken: {training_time:.2f} seconds.\n")
    test_model(model, test_data)

if __name__ == "__main__":
    main()

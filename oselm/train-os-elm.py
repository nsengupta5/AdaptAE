from oselm import OSELM
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.profiler import profile, record_function, ProfilerActivity
from sys import argv
import logging
import time
import warnings
import matplotlib.pyplot as plt
import psutil

# Constants
TRAIN_SIZE_PROP = 0.01
SEQ_SIZE_PROP = 0.99
DEFAULT_BATCH_SIZE = 20
NUM_IMAGES = 5
DEBUG = False

"""
Initialize the OSELM model
"""
def oselm_init(input_nodes, hidden_nodes):
    logging.info(f"Initializing OSELM model...")
    activation_func = 'sigmoid'
    loss_func = 'mse'
    logging.info(f"Initializing OSELM model complete.")
    return OSELM(activation_func, loss_func, input_nodes, hidden_nodes, device).to(device)

"""
Load and split the data into training, sequential and test data
param dataset: The dataset to load
param mode: The mode to load the data in
param batch_size: The batch size to use (default sample)
"""
def load_and_split_data(dataset, mode, batch_size = 1):
    logging.info(f"Loading and preparing data...")
    transform = transforms.ToTensor()
    input_nodes = 784
    hidden_nodes = 128
    if mode == "sample":
        batch_size = 1
    match dataset:
        case 'mnist':
            transform = transforms.ToTensor()
            train_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.MNIST(root = './data', train = False, download = True, transform = transform)
        case 'fashion-mnist':
            transform = transforms.ToTensor()
            train_data = datasets.FashionMNIST(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.FashionMNIST(root = './data', train = False, download = True, transform = transform)
        case 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.CIFAR100(root = './data', train = True, download = True, transform = transform)
            test_data = datasets.CIFAR100(root = './data', train = False, download = True, transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'super-tiny-imagenet':
            transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform = transform)
            test_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/test', transform = transform)
            input_nodes = 3072
            hidden_nodes = 1024
        case 'tiny-imagenet':
            transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                # Normalize each channel of the input data
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
            train_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/train', transform = transform)
            test_data = datasets.ImageFolder(root = './data/tiny-imagenet-200/test', transform = transform)
            input_nodes = 12288
            hidden_nodes = 4096
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    # Split 60% for training, 20% for sequential training
    train_size = int(TRAIN_SIZE_PROP * len(train_data))
    seq_size = int(SEQ_SIZE_PROP * len(train_data))
    train_data, seq_data = random_split(train_data, [train_size, seq_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = train_size, shuffle = True)
    seq_loader = torch.utils.data.DataLoader(seq_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle = False)
    logging.info(f"Loading and preparing data complete.")
    return train_loader, seq_loader, test_loader, input_nodes, hidden_nodes

"""
Initialize the OSELM model with the initial training data
:param model: The OSELM model
:param train_data: The initial training data
"""
def train_init(model, train_loader):
    peak_memory = 0
    process = None

    for (data, _) in train_loader:
        # Reshape the data to fit the model
        data = data.reshape(-1, model.input_shape[0]).float().to(device)
        logging.info(f"Initial training on {len(data)} samples...")
        logging.info("Train data shape: " + str(data.shape))

        # Time and CUDA memory tracking
        start_time = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            process = psutil.Process()
            peak_memory = process.memory_info().rss 

        if DEBUG:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, profile_memory=True) as prof:
            # Train the model
                with record_function("model_init_train"):
                    model.init_phase(data)

            print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        else:
            model.init_phase(data)

        end_time = time.time()

        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory)

        training_time = end_time - start_time
        title = "Initial Training Benchmarks"
        print("\n" + title)
        print("=" * len(title))
        print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
        print(f"Initial Training complete. Time taken: {training_time:.2f} seconds.\n")

        # Evaluate the model on the initial training data
        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)
        print(f"Initial training loss: {loss:.2f}")
        logging.info(f"Initial training complete")

"""
Train the OSELM model sequentially on the sequential training data
:param model: The OSELM model
:param seq_data: The sequential training data
:param mode: The mode of sequential training, either "sample" or "batch"
"""
def train_sequential(model, seq_loader, mode):
    logging.info(f"Sequential training on {len(seq_loader)} batches in {mode} mode...")
    # Metrics for each iteration
    sample_times = []
    total_loss = 0

    # Time and CUDA memory tracking
    peak_memory = 0
    process = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        process = psutil.Process()
        peak_memory = process.memory_info().rss

    start_time = time.time()
    if DEBUG:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack = True, profile_memory = True) as prof:
            with record_function("model.seq_phase"):
                for (data, _) in seq_loader:
                    # Reshape the data to fit the model
                    data = data.reshape(-1, model.input_shape[0]).float().to(device)
                    sample_start_time = time.time()

                    model.seq_phase(data, mode)

                    sample_end_time = time.time()

                    if device == "cpu":
                        current_memory = process.memory_info().rss
                        peak_memory = max(peak_memory, current_memory)

                    sample_times.append(sample_end_time - sample_start_time)

                    pred = model.predict(data)
                    loss, _ = model.evaluate(data, pred)
                    total_loss += loss.item()

        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    else:
        for (data, _) in seq_loader:
            # Reshape the data to fit the model
            data = data.reshape(-1, model.input_shape[0]).float().to(device)
            sample_start_time = time.time()

            model.seq_phase(data, mode)

            sample_end_time = time.time()

            if device == "cpu":
                current_memory = process.memory_info().rss
                peak_memory = max(peak_memory, current_memory)

            sample_times.append(sample_end_time - sample_start_time)

            pred = model.predict(data)
            loss, _ = model.evaluate(data, pred)
            total_loss += loss.item()
    
    end_time = time.time()

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()

    training_time = end_time - start_time
    sample_avg_time = sum(sample_times) / len(sample_times)

    title = "Sequential Training Benchmarks"
    print(f"\n{title}:")
    print("=" * len(title))
    print(f"Peak memory allocated during training: {peak_memory / (1024 ** 2):.2f} MB")
    print(f"Average time per sample: {sample_avg_time:.5f} seconds")
    print(f"Average Loss: {total_loss / len(seq_loader):.2f}")
    print(f"Sequential training complete. Time taken: {training_time:.2f} seconds.\n")
    logging.info(f"Sequential training complete")

"""
Test the OSELM model on the test data
:param model: The OSELM model
:param test_data: The test data
"""
def test_model(model, test_loader, dataset, mode):
    logging.info(f"Testing on {len(test_loader.dataset)} batches...")
    losses = []
    outputs = []
    saved_img = False
    for (data, _) in test_loader:
        # Reshape the data to fit the model
        data = data.reshape(-1, model.input_shape[0]).float().to(device)
        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)
        losses.append(loss.item())
        if mode == "sample" or test_loader.batch_size < NUM_IMAGES:
            outputs.append((data, pred))
        if not saved_img:
            if mode == "sample" or test_loader.batch_size < NUM_IMAGES:
                if len(outputs) > NUM_IMAGES:
                    full_data = torch.cat([data for (data, _) in outputs], dim=0)
                    full_pred = torch.cat([pred for (_, pred) in outputs], dim=0)
                    visualize_comparisons(full_data.cpu().numpy(), full_pred.cpu().detach().numpy(), dataset, test_loader.batch_size)
                    saved_img = True
            else:
                visualize_comparisons(data.cpu().numpy(), pred.cpu().detach().numpy(), dataset, test_loader.batch_size)
                saved_img = True

    loss = sum(losses) / len(losses)
    title = "Total Loss:"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Loss: {loss:.5f}\n")
    logging.info(f"Testing complete.")

"""
Exit the program with an error message of the correct usage
"""
def exit_with_error():
    print("Usage: python train-os-elm.py [mode] <batch size> [dataset] <device>")
    print("mode: sample or batch")
    print("dataset: mnist, fashion-mnist, cifar10 or cifar100")
    print("batch size: integer (only required for batch mode and defaults to 20 if not provided)")
    print("device: cpu, mps or cuda (defaults to cpu if not provided)")
    exit(1)

"""
Visualize the original and reconstructed images
:param originals: The original images
:param reconstructions: The reconstructed images
:param dataset: The dataset used
:param n: The number of images to visualize
"""
def visualize_comparisons(originals, reconstructions, dataset, batch_size, n=NUM_IMAGES):
    plt.figure(figsize=(20, 4))
    for i in range(n): # Display original images
        ax = plt.subplot(2, n, i + 1)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(originals[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(originals[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(originals[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstructed images
        ax = plt.subplot(2, n, i + 1 + n)
        if dataset in ["mnist", "fashion-mnist"]:
            plt.imshow(reconstructions[i].reshape(28, 28))
        elif dataset in ["cifar10", "cifar100", "super-tiny-imagenet"]:
            plt.imshow(reconstructions[i].reshape(3, 32, 32).transpose(1, 2, 0))
        else:
            plt.imshow(reconstructions[i].reshape(3, 64, 64).transpose(1, 2, 0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if batch_size == 1:
        plt.savefig(f"oselm/results/{dataset}-reconstructions-sample.png")
    else:
        plt.savefig(f"oselm/results/{dataset}-reconstructions-batch-{batch_size}.png")

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
Get the dataset to use (either "mnist", "fashion-mnist", "cifar10", "cifar100", or "tiny-imagenet")
:param mode: The mode of sequential training (either "sample" or "batch")
"""
def get_dataset(mode):
    arg_len = 2 if mode == "sample" else 3
    if len(argv) < arg_len + 1:
        # Default to MNIST datasets
        return "mnist"
    elif argv[arg_len] not in ["mnist", "fashion-mnist", "cifar10", "cifar100", "tiny-imagenet"]:
        exit_with_error()
    else:
        return argv[arg_len]

"""
Get the batch size to use
:param mode: The mode of sequential training (either "sample" or "batch")
"""
def get_batch_size(mode):
    if mode == "sample":
        return 1
    else:
        try:
            return int(argv[2])
        except:
            if argv[2] not in ["mnist", "fashion-mnist", "cifar10", "cifar100", "tiny-imagenet"]:
                exit_with_error()
            else:
                return DEFAULT_BATCH_SIZE

"""
Get the device to use (either "cpu", "mps" or "cuda")
"""
def get_device(mode):
    arg_len = 3 if mode == "sample" else 4
    if len(argv) < arg_len + 1:
        # Default to CPU
        return "cuda"
    elif argv[arg_len] not in ["cpu", "mps", "cuda"]:
        exit_with_error()
    else:
        return argv[arg_len]
        
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    global device
    mode = get_mode()
    dataset = get_dataset(mode)
    device = get_device(mode)
    batch_size = get_batch_size(mode)
    logging.basicConfig(level=logging.INFO)
    train_loader, seq_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(dataset, mode, batch_size)
    model = oselm_init(input_nodes, hidden_nodes)

    # Time and CUDA memory tracking
    peak_memory = 0
    process = None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        process = psutil.Process()
        peak_memory = process.memory_info().rss

    start_time = time.time()

    train_init(model, train_loader)
    train_sequential(model, seq_loader, mode)

    end_time = time.time()

    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        current_memory = process.memory_info().rss
        peak_memory = max(peak_memory, current_memory)

    training_time = end_time - start_time

    title = "Total Training Benchmarks:"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Peak memory allocated during total training: {peak_memory / (1024 ** 2):.2f} MB")
    print(f"Total training complete. Time taken: {training_time:.2f} seconds.\n")
    logging.info(f"Total training complete")
    test_model(model, test_loader, dataset, mode)

if __name__ == "__main__":
    main()

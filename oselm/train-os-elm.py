from oselm import OSELM
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import logging
import time
import warnings
import matplotlib.pyplot as plt
import psutil
import csv
import argparse

# Constants
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEQ_PROP = 0.99
NUM_IMAGES = 5

result_data = []

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
def load_and_split_data(dataset, mode, batch_size = 1, seq_prop = 0.2):
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
    seq_size = int(seq_prop * len(train_data))
    train_size = len(train_data) - seq_size
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
:param phased: Whether to ignore the initial training performance
"""
def train_init(model, train_loader, phased):
    peak_memory = 0
    process = None

    for (data, _) in train_loader:
        # Reshape the data to fit the model
        data = data.reshape(-1, model.input_shape[0]).float().to(device)
        logging.info(f"Initial training on {len(data)} samples...")
        logging.info("Train data shape: " + str(data.shape))

        # Don't reset the peak memory if we're monitoring total memory
        if phased:
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            else:
                process = psutil.Process()
                peak_memory = process.memory_info().rss 

        start_time = time.time()

        model.init_phase(data)

        end_time = time.time()

        if phased:
            if device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                current_memory = process.memory_info().rss
                peak_memory = max(peak_memory, current_memory) / (1024 ** 2)

        training_time = end_time - start_time
        title = "Initial Training Benchmarks"
        print("\n" + title)
        print("=" * len(title))
        if phased:
            print(f"Peak memory allocated during training: {peak_memory:.2f} MB")
        print(f"Time taken: {training_time:.2f} seconds.")

        # Evaluate the model on the initial training data
        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)
        print(f"Initial training loss: {loss:.2f}")

        # Saving results
        if phased:
            result_data.append(training_time)
            result_data.append(round(peak_memory, 2))
            result_data.append(float(str(f"{loss:.3f}")))

        logging.info(f"Initial training complete")

"""
Train the OSELM model sequentially on the sequential training data
:param model: The OSELM model
:param seq_data: The sequential training data
:param mode: The mode of sequential training, either "sample" or "batch"
:param phased: Whether to ignore sequential training performance
"""
def train_sequential(model, seq_loader, mode, phased):
    logging.info(f"Sequential training on {len(seq_loader)} batches in {mode} mode...")
    # Metrics for each iteration
    sample_times = []
    total_loss = 0

    # Time and CUDA memory tracking
    peak_memory = 0
    process = None
    if phased:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            process = psutil.Process()
            peak_memory = process.memory_info().rss

    start_time = time.time()
    for (data, _) in seq_loader:
        # Reshape the data to fit the model
        data = data.reshape(-1, model.input_shape[0]).float().to(device)
        sample_start_time = time.time()

        model.seq_phase(data, mode)

        sample_end_time = time.time()

        if phased:
            if device == "cpu":
                current_memory = process.memory_info().rss
                peak_memory = max(peak_memory, current_memory)

        sample_times.append(sample_end_time - sample_start_time)

        pred = model.predict(data)
        loss, _ = model.evaluate(data, pred)
        total_loss += loss.item()
    
    end_time = time.time()

    if phased:
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)

    training_time = end_time - start_time
    sample_avg_time = sum(sample_times) / len(sample_times)

    title = "Sequential Training Benchmarks"
    print(f"\n{title}:")
    print("=" * len(title))
    if phased:
        print(f"Peak memory allocated during training: {peak_memory:.2f} MB")
    print(f"Average time per sample: {sample_avg_time:.5f} seconds")
    print(f"Average Loss: {total_loss / len(seq_loader):.2f}")
    print(f"Time taken: {training_time:.2f} seconds.")

    # Saving results
    if phased:
        result_data.append(training_time)
        result_data.append(round(peak_memory, 2))
        result_data.append(float(str(f"{(total_loss / len(seq_loader)):3f}")))

    logging.info(f"Sequential training complete")

"""
Train the model
:param model: The model to train
:param train_loader: The training data loader
:param seq_loader: The sequential training data loader
:param mode: The mode of sequential training
:param device: The device to use
:param phased: Whether to monitor the total performance of the model
"""
def train_model(model, train_loader, seq_loader, mode, device, phased):
    peak_memory = 0
    process = None
    if not phased:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        else:
            process = psutil.Process()
            peak_memory = process.memory_info().rss

    start_time = time.time()
    train_init(model, train_loader, phased)
    train_sequential(model, seq_loader, mode, phased)
    end_time = time.time()

    if not phased:
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            current_memory = process.memory_info().rss
            peak_memory = max(peak_memory, current_memory) / (1024 ** 2)

    training_time = end_time - start_time
    title = "Total Training Benchmarks"
    print("\n" + title)
    print("=" * len(title))
    if not phased:
        print(f"Peak memory allocated during training: {peak_memory:.2f} MB")
        result_data.append(training_time)
        result_data.append(round(peak_memory, 2))
    print(f"Time taken: {training_time:.2f} seconds.")

"""
Test the OSELM model on the test data
:param model: The OSELM model
:param test_data: The test data
"""
def test_model(model, test_loader, dataset, gen_imgs):
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
        if test_loader.batch_size < NUM_IMAGES:
            outputs.append((data, pred))
        if gen_imgs:
            if not saved_img:
                if test_loader.batch_size < NUM_IMAGES:
                    if len(outputs) > NUM_IMAGES:
                        full_data = torch.cat([data for (data, _) in outputs], dim=0)
                        full_pred = torch.cat([pred for (_, pred) in outputs], dim=0)
                        visualize_comparisons(
                            full_data.cpu().numpy(), 
                            full_pred.cpu().detach().numpy(), 
                            dataset, 
                            test_loader.batch_size
                        )
                        saved_img = True
                else:
                    visualize_comparisons(
                        data.cpu().numpy(),
                        pred.cpu().detach().numpy(),
                        dataset,
                        test_loader.batch_size
                    )
                    saved_img = True

    loss = sum(losses) / len(losses)
    title = "Total Loss:"
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Loss: {loss:.5f}\n")

    # Saving results
    result_data.append(float(str(f"{loss:.5f}")))

    logging.info(f"Testing complete.")

"""
Exit the program with an error message of the correct usage
"""
def exit_with_error(msg, parser):
    logging.error(msg)
    parser.print_help()
    exit(1)

"""
Visualize the original and reconstructed images
:param originals: The original images
:param reconstructions: The reconstructed images
:param dataset: The dataset used
:param n: The number of images to visualize
"""
def visualize_comparisons(originals, reconstructions, dataset, batch_size, n=NUM_IMAGES):
    logging.info(f"Generating {n} images...")
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

    logging.info(f"Saving images to oselm/results/ ...")
    if batch_size == 1:
        plt.savefig(f"oselm/results/{dataset}-reconstructions-sample.png")
    else:
        plt.savefig(f"oselm/results/{dataset}-reconstructions-batch-{batch_size}.png")

"""
Save the results to a CSV file
"""
def save_result_data(dataset, phased, result_strategy):
    target_dir = "phased" if phased else "total"
    with open (f'oselm/data/{target_dir}/{result_strategy}_{dataset}_performance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result_data)

""" 
Get the arguments from the command line
"""
def get_args():
    parser = argparse.ArgumentParser(description="Training an OS-ELM model")
    # Define the arguments
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["sample", "batch"],
        required=True,
        help="The mode of sequential training (either 'sample' or 'batch')"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "fashion-mnist", "cifar10", "cifar100", "super-tiny-imagenet", "tiny-imagenet"],
        required=True,
        help="The dataset to use (either 'mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'super-tiny-imagenet' or 'tiny-imagenet')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The batch size to use. Must be provided if using batch mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "mps", "cuda"],
        default="cuda",
        help="The device to use (either 'cpu', 'mps' or 'cuda')"
    )
    parser.add_argument(
        "--seq-prop",
        type=float,
        default=DEFAULT_SEQ_PROP,
        help="The sequential training data proportion. Must be between 0.01 and 0.99 inclusive"
    )
    parser.add_argument(
        "--generate-imgs",
        action="store_true",
        help="Whether to generate images of the reconstructions"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Whether to save the results to a CSV file"
    )
    parser.add_argument(
        "--phased",
        action="store_true",
        help="Whether to monitor and save phased or total performance results"
    )
    parser.add_argument(
        "--result-strategy",
        type=str,
        choices=["batch-size", "seq-prop", "total"],
        help="If saving results, the independent variable to vary when saving results"
    )

    # Parse the arguments
    args = parser.parse_args()
    mode = args.mode
    dataset = args.dataset
    device = args.device
    gen_imgs = args.generate_imgs
    save_results = args.save_results
    phased = args.phased
    result_strategy = args.result_strategy

    # Assume sample mode if no mode is specified
    batch_size = 1
    if args.mode == "batch": 
        if args.batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE
        else:
            batch_size = args.batch_size
    else:
        if args.batch_size is not None:
            # Batch size is not used in sample mode
            exit_with_error("Batch size is not used in sample mode", parser)

    seq_prop = DEFAULT_SEQ_PROP
    if args.seq_prop is not None:
        if args.seq_prop <= 0 or args.seq_prop >= 1:
            # Sequential proportion must be between 0 and 1
            exit_with_error("Sequential proportion must be between 0 and 1", parser)
        else:
            seq_prop = args.seq_prop

    if args.save_results:
        if args.result_strategy is None:
            # Must specify a result strategy if saving result
            exit_with_error("Must specify a result strategy if saving results", parser)

    return mode, dataset, batch_size, device, seq_prop, gen_imgs, save_results, phased, result_strategy

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    global device
    global save_results
    mode, dataset, batch_size, device, seq_prop, gen_imgs, save_results, phased, result_strategy = get_args()
    if save_results:
        match result_strategy:
            case "batch-size":
                result_data.append(batch_size)
            case "seq-prop":
                result_data.append(seq_prop)
            case "total":
                result_data.append(batch_size)
                result_data.append(seq_prop)
    logging.basicConfig(level=logging.INFO)
    train_loader, seq_loader, test_loader, input_nodes, hidden_nodes = load_and_split_data(dataset, mode, batch_size, seq_prop)
    model = oselm_init(input_nodes, hidden_nodes)
    train_model(model, train_loader, seq_loader, mode, device, phased)
    test_model(model, test_loader, dataset, gen_imgs)
    if save_results:
        save_result_data(dataset, phased, result_strategy)

if __name__ == "__main__":
    main()

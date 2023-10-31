from oselm import OSELM, assert_cond
from torchvision import datasets, transforms
from torch.backends import mps
from torch import cuda, clamp, set_printoptions
from torch.utils.data import random_split
from sys import argv
import logging

# Constants
TRAIN_SIZE_PROP = 0.6
SEQ_SIZE_PROP = 0.2
TEST_SIZE_PROP = 0.2
BATCH_SIZE = 64
INPUT_NODES = 784
HIDDEN_NODES = 128
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
def oselm_init():
    activation_func = 'sigmoid'
    loss_func = 'mse'
    return OSELM(activation_func, loss_func, INPUT_NODES, HIDDEN_NODES).to(DEVICE)

"""
Load and split the MNIST data into training, sequential and test data
"""
def load_and_split_data():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)

    # Split 60% for training, 20% for sequential training and 20% for testing
    train_size = int(0.6 * len(mnist_data))
    seq_size = int(0.2 * len(mnist_data))
    test_size = len(mnist_data) - train_size - seq_size
    train_data, seq_data, test_data = random_split(mnist_data, [train_size, seq_size, test_size])
    return train_data, seq_data, test_data

"""
Initialize the OSELM model with the initial training data
:param model: The OSELM model
:param train_data: The initial training data
"""
def train_init(model, train_data):
    # Assert that the initial training data is of the correct shape
    data = train_data.dataset.data.view(-1, 784).float().to(DEVICE)
    assert_cond(data.shape[0] == len(train_data.dataset), "Train data shape mismatch")
    logging.info(f"Initial training on {len(data)} samples...")
    logging.info("Train data shape: " + str(data.shape))
    model.init_phase(data)
    logging.info(f"Initial training complete.")

"""
Train the OSELM model sequentially on the sequential training data
:param model: The OSELM model
:param seq_data: The sequential training data
:param mode: The mode of sequential training, either "sample" or "batch"
"""
def train_sequential(model, seq_data, mode):
    logging.info(f"Sequential training on {len(seq_data.dataset)} samples in {mode} mode...")
    data = seq_data.dataset.data.view(-1, 784).float().to(DEVICE)
    assert_cond(data.shape[0] == len(seq_data.dataset), "Sequential data shape mismatch")
    logging.info("Sequential data shape: " + str(data.shape))
    if mode == "sample":
        for image in data:
            model.seq_phase(image, mode)
    else:
        for i in range(0, len(data), BATCH_SIZE):
            images = data[i:i+BATCH_SIZE]
            model.seq_phase(images, mode)
    logging.info(f"Sequential training complete.")

"""
Test the OSELM model on the test data
:param model: The OSELM model
:param test_data: The test data
"""
def test_model(model, test_data):
    logging.info(f"Testing on {len(test_data.dataset)} samples...")
    set_printoptions(sci_mode=False)
    data = test_data.dataset.data.view(-1, 784).float().to(DEVICE)
    assert_cond(data.shape[0] == len(test_data.dataset), "Test data shape mismatch")
    pred = model.predict(data)
    pred = clamp(pred, min=0).round().int()
    print(pred[0])
    print(data[0])
    loss, accuracy = model.evaluate(data, pred)
    print(f"Loss: {loss.item():.2f}")
    print(f"Accuracy: {accuracy.item():.2f}%")

def exit_with_error():
    print("Usage: train-os-elm.py <mode>")
    print("mode: sample or batch")
    exit(1)

def main():
    # Parse command line arguments
    if len(argv) == 1:
        # Default to sample mode
        mode = "sample"
    elif len(argv) == 2:
        if argv[1] not in ["sample", "batch"]:
            exit_with_error()
        else:
            mode = argv[1]
    else:
        exit_with_error()

    logging.basicConfig(level=logging.INFO)
    train_data, seq_data, test_data = load_and_split_data()
    model = oselm_init()
    train_init(model, train_data)
    train_sequential(model, seq_data, mode)
    test_model(model, test_data)

if __name__ == "__main__":
    main()

from oselm import OSELM
from torchvision import datasets, transforms
from torch.backends import mps
from torch import cuda
from torch.utils.data import random_split

TRAIN_SIZE_PROP = 0.6
SEQ_SIZE_PROP = 0.2
TEST_SIZE_PROP = 0.2
BATCH_SIZE = 64

# Get CPU, GPU or MPS Device for training
device = (
    "cuda"
    if cuda.is_available()
    else "mps"
    if mps.is_available()
    else "cpu"
)

def oselm_init():
    activation_func = 'sigmoid'
    loss_func = 'mse'
    n_hidden_nodes = 128
    n_input_nodes = 784
    return OSELM(activation_func, loss_func, n_input_nodes, n_hidden_nodes).to(device)

# Data Loading and Splitting
def load_and_split_data():
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST(root = './data', train = True, download = True, transform = transform)
    train_size = int(0.6 * len(mnist_data))
    seq_size = int(0.2 * len(mnist_data))
    test_size = len(mnist_data) - train_size - seq_size
    train_data, seq_data, test_data = random_split(mnist_data, [train_size, seq_size, test_size])
    return train_data, seq_data, test_data

def train_model(model, train_data, seq_data, mode):
    data = train_data.dataset.data.view(-1, 784).float().to(device)
    model.init_phase(data)

#     if mode == "sample":
#         for i in range(len(seq_data.dataset)):
#             image, _ = seq_data.dataset[i]
#             sample = image.view(-1, 784).float().to(device)
#             model.seq_phase(sample, mode)
#     else:
#         for i in range(0, len(seq_data.dataset), BATCH_SIZE):
#             images, _ = seq_data.dataset[0][i:i+BATCH_SIZE]
#             batch = images.view(-1, 784).float().to(device)
#             model.seq_phase(batch, mode)

def test_model(model, test_data):
    data = test_data.dataset.data.view(-1, 784).float().to(device)
    pred = model.predict(data)
    loss, accuracy = model.evaluate(data, pred)
    print(f"Loss: {loss.item():.2f}")
    print(f"Accuracy: {accuracy.item():.2f}%")

def main():
    train_data, seq_data, test_data = load_and_split_data()
    model = oselm_init()
    train_model(model, train_data, seq_data, mode = "sample")
    test_model(model, test_data)
    
if __name__ == "__main__":
    main()

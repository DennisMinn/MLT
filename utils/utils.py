from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision import transforms

def load_MNIST(data_path):
    mnist_train = MNIST(data_path,
            download=True,
            train=True,
            transform=transforms.ToTensor())

    mnist_test = MNIST(data_path,
            download=True,
            train=False,
            transform=transforms.ToTensor())

    return mnist_train, mnist_test

def load_fashion(data_path):
    fashion_train = FashionMNIST(data_path,
            download=True,
            train=True,
            transform=transforms.ToTensor())

    fashion_test = FashionMNIST(data_path,
            download=True,
            train=False,
            transform=transforms.ToTensor())

    return fashion_train, fashion_test


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Subset
def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root = './data', train = True , download=True , transform=transform)

    trainloader = DataLoader(trainset , batch_size = 64,shuffle = True)

    return trainset,trainloader


def load_cifar10_test(batch_size: int = 64):
    """Load CIFAR-10 test dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return testset, testloader

def Dirichlet_partition(dataset , num_clients = 5 , alpha = 0.5):
    """
    Partition the dataset indices into `num_clients` parts using Dirichlet distribution per class.

    Returns a list of lists, where each inner list contains indices for one client.
    """
    num_classes = len(set([label for _, label in dataset]))
    class_indices = [[] for _ in range(num_classes)]

    for idx , (_,label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(class_indices[c])
        proportions = np.random.dirichlet([alpha]*num_clients)
        proportions = (np.cumsum(proportions)*len(class_indices[c])).astype(int)[:-1]
        splits = np.split(class_indices[c],proportions)
        for i,split in enumerate(splits):
            client_indices[i].extend(split.tolist())

    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
    
    return client_indices

def create_client_datasets(trainset: datasets.CIFAR10, client_indices):
    """Create client datasets from indices."""
    return [Subset(trainset, idx_list) for idx_list in client_indices]


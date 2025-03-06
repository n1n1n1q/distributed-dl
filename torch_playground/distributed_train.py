import torch
import torchvision as tv
import argparse
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from tqdm import tqdm
from random import Random

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def partition_dataset():
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    init_dataset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset = torch.utils.data.Subset(init_dataset, random.sample(range(len(init_dataset)), min(args.sample_size, len(init_dataset))))
    sz = dist.get_world_size()
    bsz = 128 // sz
    partition_sizes = [1. / sz for _ in range(sz)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(dataset, batch_size=bsz, shuffle=True)
    return train_set, bsz

def train(model, optimizer, criterion, device):
    print(f"Training the model.")
    model.train()
    train_set, bsz = partition_dataset()
    total_loss = 0
    correct = 0
    counter = 0
    for i, data in tqdm(enumerate(train_set)):
        counter += 1
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        correct += (pred == labels).sum().item()
        loss.backward()
        average_gradients(model)
        optimizer.step()
    loss = total_loss / counter
    accuracy = 100 * correct / len(train_set.dataset)
    return loss, accuracy

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def worker(rank, size):
    print(f"Process {rank} initialized in a group of {size}")

def init_process(rank, size, fn, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Resnet tuning on a CIFAR dataset")
    parser.add_argument("n_epochs", nargs="?", default = 10, type = int)
    parser.add_argument("sample_size", nargs="?", default = 1000, type = int)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resnet = torch.load("model.pth")
    except:
        resnet = tv.models.resnet18(pretrained=True)
    init_process(0, 1, worker)
    model = resnet.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(args.n_epochs):
        loss, accuracy = train(model, optimizer, criterion, device)
        print(f"Epoch {epoch}: Loss: {loss}, Accuracy: {accuracy}")
    for grad in model.parameters():
        print(grad.grad)

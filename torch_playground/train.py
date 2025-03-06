import torch
import torchvision as tv
import argparse
import random
from tqdm import tqdm

def train(model, train_loader, optimizer, criterion, device):
    print(f"Training the model. {len(train_loader)} batches")
    model.train()
    total_loss = 0
    correct = 0
    counter = 0
    for i, data in tqdm(enumerate(train_loader)):
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
        optimizer.step()
    loss = total_loss / counter
    accuracy = 100 * correct / len(train_loader.dataset)
    return loss, accuracy

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
    # resnet.fc = torch.nn.Linear(512, 10)
    model = resnet.to(device)
    transform = tv.transforms.Compose([
        # tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    init_dataset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # init_dataset = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    dataset = torch.utils.data.Subset(init_dataset, random.sample(range(len(init_dataset)), min(args.sample_size, len(init_dataset))))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(args.n_epochs):
        loss, accuracy = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: Loss: {loss}, Accuracy: {accuracy}")
    for grad in model.parameters():
        print(grad.grad)
    torch.save(model, "model.pth")

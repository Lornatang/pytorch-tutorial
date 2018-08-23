"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: cnn.py
# time: 2018/7/29 15:28
# license: MIT
"""

import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../../data/mnist',
                    help="""image path. Default='../../data/mnist'.""")
parser.add_argument('--epochs', type=int, default=20,
                    help="""num epochs. Default=20""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""0 ~ 9,. Default=10""")
parser.add_argument('--batch_size', type=int, default=256,
                    help="""batch size. Default=256""")
parser.add_argument('--display_step', type=float, default=256)
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
args = parser.parse_args()

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root=args.path,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=args.path,
                                          train=False,
                                          transform=transforms.ToTensor(),
                                          download=False)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)


class CNN(nn.Module):
    def __init__(self, category=10):
        super(CNN, self).__init__()
        self.category = category
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, args.num_classes)

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        out = layer2.reshape(layer2.size(0), -1)
        out = self.fc(out)

        return out


# Load model
model = CNN(args.num_classes).to(device)
# Correct
correct = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


# Train
def train():
    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (img, labs) in enumerate(train_loader):
            img = img.to(device)
            labs = labs.to(device)

            # Forward pass
            pred = model(img)
            loss = correct(pred, labs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch}/{args.max_epochs}] loss {loss.item():.8f}")
    evalution()

    torch.save(model, 'model.ckpt')


def evalution():
    model.eval()
    with torch.no_grad():
        correct_prediction = 0
        for img, labs in test_loader:
            img = img.to(device)
            labs = labs.to(device)

            output = model(img)
            _, prediction = torch.max(output.data, 1)
            correct_prediction += (prediction == labs).sum().item()

        print(f"Acc {100 * correct_prediction / 10000:.2f}%")


if __name__ == '__main__':
    train()

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch
from cifar_dataset import CIFAR10

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BiggerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, kernel_size=(3,3), padding="same", out_channels=32)
        self.conv2 = nn.Conv2d(in_channels=32, kernel_size=(3,3), padding="same", out_channels=64)
        self.conv3 = nn.Conv2d(in_channels=64, kernel_size=(3,3), padding="same", out_channels=128)

        self.norm1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=0)
        
    def forward(self, x):
        x = self.norm1(self.conv1(x))
        x = self.norm1(self.conv1(x))
        x = self.dropout(self)




if __name__ == "__main__":
    cifar_data = CIFAR10() 
    cifar_data.generate()

    dat_iter = iter(cifar_data.trainloader)
    image_input, targets = next(dat_iter)
    netbig = BiggerNet()
    print(image_input.shape) # [4, 3, 32, 32]) / [no of images, channels, H, W]
    print(netbig.pool(netbig.conv1(image_input)).shape) # [4, 32, 32, 32]

    if os.getenv("TRAIN"):
        print("[cnn_net] training net on train dataset.")
        assert cifar_data.trainloader != None
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        Epochs = 2
        for epoch in range(Epochs):
            running_loss = 0.0
            for i, data in enumerate(cifar_data.trainloader, 0):
                inputs, targets = data
                # zero out the gradients
                optimizer.zero_grad()
                # forward + backward propogation
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # update weights based on gradients
                optimizer.step()
                running_loss += loss.item()
                # stats every 2000 mini-batches
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0
        print("[cnn_net] finished training and saved to artifacts!")
        torch.save(net.state_dict(), "artficats/cifar_net.pth")

    if os.getenv("TEST"):
        print("[cnn_net] testing on test dataset.")
        net = Net()
        net.load_state_dict(torch.load("artficats/cifar_net.pth", weights_only=True))
        print(net.eval())
        correct = 0
        total = 0
        with torch.no_grad():
            for data in cifar_data.testloader:
                images, target = data
                outputs = net(images)
                # pick with max prob
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # 
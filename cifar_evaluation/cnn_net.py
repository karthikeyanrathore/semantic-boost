import os
import time

import torch
import torch.nn.functional as F
from torch import nn, optim

__all__ = ["Net", "CNNTrainer", "get_default_device"]


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 10)
        self.features = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.bn3(self.conv4(x)))
        x = self.pool(x)
        self.features = x
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device=None,
        lr=0.001,
        momentum=0.9,
        log_interval=200,
    ):
        self.device = device or get_default_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        self.log_interval = log_interval
        self.history = {"train_loss": [], "val_loss": []}

    def train(self, epochs):
        start = time.time()
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if self.log_interval and batch_idx % self.log_interval == 0:
                    avg = total_loss / batch_idx
                    print(f"epoch {epoch} step {batch_idx}: running loss {avg:.3f}")
            train_loss = total_loss / len(self.train_loader)
            val_loss = self._evaluate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(
                f"epoch {epoch} done: train loss {train_loss:.4f}, "
                f"val loss {val_loss:.4f}"
            )
        print(f"total training time: {time.time() - start:.1f}s")

    def _evaluate(self):
        self.model.eval()
        if len(self.val_loader) == 0:
            return 0.0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                val_loss += self.criterion(outputs, labels).item()
        return val_loss / len(self.val_loader)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

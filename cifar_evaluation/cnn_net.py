import os
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt

__all__ = ["Net", "CNNNetTrainer", "get_default_device", "compute_model_accuracy"]

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
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        self.features = x
        x = self.fc2(x)
        return x

def build_trainer_config(*args, **kwargs):
    if len(args) > 7:
        raise TypeError("CNNNetTrainer expected at most 7 positional arguments")
    param_keys = [
        "model",
        "train_loader",
        "val_loader",
        "device",
        "lr",
        "momentum",
        "log_interval",
    ]
    trainer_config = {}
    trainer_config["model"] = None
    trainer_config["train_loader"] = None
    trainer_config["val_loader"] = None
    trainer_config["device"] = None
    trainer_config["lr"] = 0.001
    trainer_config["momentum"] = 0.9
    trainer_config["log_interval"] = 200
    for key, value in zip(param_keys, args):
        trainer_config[key] = value
    trainer_config.update(kwargs)
    if (trainer_config["model"] is None or trainer_config["train_loader"] is None or trainer_config["val_loader"] is None):
        raise TypeError("CNNNetTrainer requires model, train_loader, and val_loader")
    return trainer_config

def compute_model_accuracy(model_path, test_loader):
    print(f"test_loader: {len(test_loader)} batches")
    device = get_default_device()
    model = Net()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    loss = nn.CrossEntropyLoss()
    num_correct, num_total = 0.0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device) 
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for truth, predicted in zip(labels, predictions):
                if truth == predicted:
                    num_correct += 1.0
                num_total += 1.0
    return num_correct / num_total

class CNNNetTrainer:
    def __init__(self, *args, **kwargs):
        trainer_config = build_trainer_config(*args, **kwargs)
        self.device = trainer_config["device"] or get_default_device()
        self.model = trainer_config["model"].to(self.device)
        self.train_loader = trainer_config["train_loader"]
        self.val_loader = trainer_config["val_loader"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=trainer_config["lr"],
            momentum=trainer_config["momentum"],
        )
        self.log_interval = trainer_config["log_interval"]
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
                    avg_batch_loss = total_loss / batch_idx
                    print(f"epoch {epoch} step {batch_idx}: running loss {avg_batch_loss:.3f}")
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

    def plot_curve(self, out_path="loss_curve.png"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.history["train_loss"], label="trainingloss", color="green")
        ax.plot(self.history["val_loss"], label="validationloss", color="blue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

import argparse
import os

from cifar_evaluation.cifar_dataset import CIFAR10Data
from cifar_evaluation.cnn_net import CNNNetTrainer, Net, get_default_device, compute_model_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 CNN")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="SGD learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--log-interval", type=int, default=200, help="Steps between status prints"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.8,
        help="Portion of test split used for validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting validation/test"
    )
    parser.add_argument(
        "--model-path",
        default="cifar_net_train.pth",
        help="Path where the trained checkpoint will be saved",
    )
    parser.add_argument("--root", default="./data", help="Dataset root directory")
    return parser.parse_args()


def main():
    args = parse_args()
    data = CIFAR10Data(
        root=args.root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader, val_loader, test_loader = data.create_loaders()
    if os.getenv("TRAINING"):
        model = Net()
        device = get_default_device()
        cnn_trainer = CNNNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=args.learning_rate,
            momentum=args.momentum,
            log_interval=args.log_interval,
        )
        cnn_trainer.train(args.epochs)
        os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
        cnn_trainer.save(args.model_path)
        cnn_trainer.plot_curve()
        print("training finished", f"checkpoint saved at {args.model_path}")
    test_accuracy = compute_model_accuracy(args.model_path, test_loader)
    print(f"Accuracy on test dataset: {test_accuracy}")

if __name__ == "__main__":
    main()

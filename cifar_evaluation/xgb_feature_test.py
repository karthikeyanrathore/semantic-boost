import argparse
import os

import numpy as np
import torch
import xgboost as xgb

from .cifar_dataset import CIFAR10Data
from .cnn_net import Net, get_default_device


def extract_features(loader, model, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            model(images)
            feature_map = model.features
            if feature_map is None:
                raise RuntimeError("features were not populated by the CNN")
            feature_map = feature_map.reshape(images.size(0), -1).cpu()
            features.append(feature_map)
            labels.append(targets)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train XGBoost on CNN features from CIFAR-10"
    )
    parser.add_argument(
        "--model-path",
        default="cifar_net_train.pth",
        help="Path to a trained CNN checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for loading CIFAR-10",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.8,
        help="Percentage of CIFAR-10 test split reserved for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to split validation/test",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=100,
        help="Boosting rounds for the XGBoost classifier",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"CNN checkpoint not found at {args.model_path}. Train the model first."
        )

    data = CIFAR10Data(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader, _, test_loader = data.create_loaders()
    device = get_default_device()
    model = Net()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    train_features, train_labels = extract_features(train_loader, model, device)
    test_features, test_labels = extract_features(test_loader, model, device)

    print(
        "Features->",
        f"train {train_features.shape}, test {test_features.shape}",
    )

    print(" -- Traing XGBoost Model from CNN feature extraction --")
    dtrain = xgb.DMatrix(train_features, label=train_labels)
    dtest = xgb.DMatrix(test_features, label=test_labels)
    xgb_params = {}
    xgb_params["objective"] = "multi:softmax"
    xgb_params["num_class"] = 10
    xgb_params["eval_metric"] = "mlogloss"
    xgb_params["max_depth"] = 6
    booster = xgb.train(xgb_params, dtrain, args.num_rounds)

    train_predictions = booster.predict(dtrain)
    test_predictions = booster.predict(dtest)
    train_accuracy = float(np.mean(train_predictions == train_labels)) * 100
    test_accuracy = float(np.mean(test_predictions == test_labels)) * 100

    print(f"XGB on train features accuracy: {train_accuracy:.2f}%")
    print(f"XGB on test features accuracy : {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()

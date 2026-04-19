from src import utils
from src import engine
from src import swinxception
from src import datasets

import torch
import torch.nn as nn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = engine.load_swinxception_model()

    _, _, loader, _ = datasets.load_datasets()

    loss, acc = engine.validate(model, loader, nn.CrossEntropyLoss(), torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Accuracy: {acc:.2f}")
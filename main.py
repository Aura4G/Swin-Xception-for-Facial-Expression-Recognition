from src import utils
from src import engine
from src import swinxception
from src import datasets

import torch
from src.swinxception import SwinXception

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    model = SwinXception(num_classes=7).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters in model: {total_params}")
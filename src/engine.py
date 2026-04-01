import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from . import swinxception

PATH = "model_checkpoints/latest.pth"

def train_one_epoch(model, data_loader, criterion, optimiser, device):
    """Makes predictions for every image in the training set for one epoch, calculates loss and accuracy, and backpropagates."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimiser.zero_grad()
        
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct * 100. / total
    return epoch_loss, epoch_acc


def validate(model, data_loader, criterion, device):
    """Makes predictions for every image in the validation/test set for one epoch, and calculates loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(data_loader)
    val_acc = correct * 100. / total
    
    return val_loss, val_acc

def training_loop(model, train_loader, val_loader, criterion, optimiser, scheduler, device, start_epoch, epochs=100):
    """
    Runs the train and validate functions for every epoch left in an unfinished model/from the start. If the
    model is starting from scratch, the model freezes all of its parameters except the mlp head to warm up the
    head for the first 3 epochs.
    """

    if start_epoch == 0:
        # Freeze backbone, train only head (Epochs 1-3)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True
        
        print("Warming up head (3 Epochs)...")
        # Use AdamW for quick head alignment
        warmup_optim = AdamW(model.head.parameters(), lr=3e-3)
    
        for epoch in range(3):
            model.train()
            for images, labels in tqdm(train_loader, desc=f"Warmup {epoch+1}"):
                images, labels = images.to(device), labels.to(device)
                warmup_optim.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                warmup_optim.step()

        # Unfreeze and train all
        print("Fine-tuning entire model...")
        for param in model.parameters():
            param.requires_grad = True

    for epoch in range(start_epoch, epochs):
        print("="*60)
        print(f"Epoch {epoch+1}/{epochs}")
    
        print("Training on RAF-DB...")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimiser, device)

        print("Validating on RAF-DB...")
        val_loss, val_acc = validate(model, val_loader, criterion, device)
    
        scheduler.step()

        print(f"RAF-DB Training    | Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%")
        print(f"RAF-DB Validation  | Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%")

        # Checkpoint the model and training information after each epoch in case of interruption
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }

        # Save the state dictionary to our checkpoint path
        torch.save(checkpoint, PATH)
    
    print("="*60)

    # Save the most optimal model to a state dictionary.
    torch.save(model.state_dict(), 'swin_xception_baseline.pth')

    return model


def retrain_mlp_head(model, features, labels, device, epochs=20, batch_size=128, lr=1e-3, weight_decay=1e-3):
    """Freezes the parameters of the model to retrain the head on the newly SMOTE'd dataset"""

    # Freeze all params in SwinXception backbone
    for param in model.patch_embed.parameters():
        param.requires_grad = False
    for stage in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in stage:
            for param in block.parameters():
                param.requires_grad = False

    for merge in [model.merge1, model.merge2, model.merge3]:
        for param in merge.parameters():
            param.requires_grad = False

    # Safe for the model's head
    for param in model.head.parameters():
        param.requires_grad = True

    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)
    dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_features, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimiser.zero_grad()

            outputs = model.head(batch_features)

            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimiser.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct * 100. / total

        print(f"Epoch {epoch+1}/{epochs}   Loss: {epoch_loss}   Accuracy: {epoch_acc}")

    return model

def load_swinxception_model(file_name='swin_xception_final.pth', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = swinxception.SwinXception(num_classes=7).to(device)

    swin_xception_final = torch.load(file_name, map_location=device)

    model.load_state_dict(swin_xception_final)

    return model
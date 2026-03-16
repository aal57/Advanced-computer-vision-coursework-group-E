import torch
from torch import nn
from backbone import new_backbone
import numpy as np
import torch.optim as optim
import dataset
from torch.utils.data import DataLoader
# TODO save your best model and store it at './models/d1.pth'

RUN_ID = "baseline"

test_path = 'D-tasks/data/test.pkl'
train_path = 'D-tasks/data/train.pkl'
meta_path = 'D-tasks/data/meta.pkl'

class FineGrainedClassifierCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 100,
        hidden_channels: int = 512,
        dropout: float = 0.25,
    ):
        super().__init__()

        # Backbone f
        self.f = new_backbone()  # returns embedding vector for each image

        with torch.no_grad():
            self.f.eval()
            dummy = torch.zeros(1, 3, 32, 32)
            feat_dim = self.f(dummy).shape[1]
        self.f.train()

        fmap_size = 6
        self.fmap_size = fmap_size

        self.reshape_channels = hidden_channels

        self.proj = nn.Linear(feat_dim, hidden_channels * fmap_size * fmap_size)

        # Head h
        self.h = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        # Output classes
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        z = self.f(x)  # (B, feat_dim)

        z = self.proj(z) # (B, C*H*W)
        z = z.view(z.size(0), self.reshape_channels,
                   self.fmap_size, self.fmap_size) # (B, C, H, W)

        z = self.h(z) # (B, C, 1, 1)                                       
        z = z.view(z.size(0), -1) # (B, C)                           

        logits = self.fc(z)
        return logits


def accuracy_top1(logits, targets):
    """
    Computes top-1 accuracy
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
    return correct / total

def mixup_data(x, y, alpha=1.0):
    """
    Data Augmentation method mix up
    Overlaying 2 images into 1 image
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, fine_labels, coarse_labels in dataloader:
        images = images.to(device)
        fine_labels = fine_labels.to(device)

        # Forward
        mixed_images, y_a, y_b, lam = mixup_data(images, fine_labels, 0.2)
        logits = model(mixed_images)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        batch_size = fine_labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_top1(logits, fine_labels) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, fine_labels,coarse_labels in dataloader:
            images = images.to(device)
            fine_labels = fine_labels.to(device)

            logits = model(images)
            loss = criterion(logits, fine_labels)

            batch_size = fine_labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_top1(logits, fine_labels) * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def main():
    # Hyperparameters 
    num_epochs = 250
    batch_size = 16
    lr_backbone = 5e-4   
    lr_head = 2e-3       
    weight_decay = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_dataset, test_dataset = dataset.create_dataset(train_path, test_path,meta_path)

    train_loader = DataLoader(
        dataset =train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset= test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = FineGrainedClassifierCNN()
    model.to(device)

    for p in model.f.parameters():
        p.requires_grad = True

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(
    [
        {"params": model.f.parameters(), "lr": lr_backbone},
        {"params": model.h.parameters(), "lr": lr_head},
    ],
    weight_decay=weight_decay,
    betas=(0.9, 0.97),)

    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[175, 225], gamma=0.15)

    # Lists for data logging
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_test_acc = 0.0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # Save best model based on test accuracy
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "D-tasks\models\d1.pth")

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )

    # Store training logs
    run_id = RUN_ID
    log_path = f"D-tasks/logs/training_{run_id}.npz"

    np.savez(
        log_path,
        train_losses=train_losses,
        train_accs=train_accs,
        test_losses=test_losses,
        test_accs=test_accs,
    )

    print(f"Best Test Acc: {best_test_acc*100:.2f}%")

def prepare_test():
    # TODO: Create an instance of your model here. Load the pre-trained weights and return your model.
    #  Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 100), where B is the batch size
    #  and 100 is the number of classes. The output of your model must be the prediction of your classifier,
    #  providing a score for each class, for each image in input
    

    model = FineGrainedClassifierCNN()

    # do not edit from here downwards
    weights_path = 'D-tasks\models\d1.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model



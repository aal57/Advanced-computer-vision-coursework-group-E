import torch
from torch import nn
from backbone import new_backbone
import numpy as np
import torch.optim as optim
import dataset
from torch.utils.data import DataLoader
# TODO save your best model and store it at './models/d3.pth'

RUN_ID = "baseline"

test_path = 'D-tasks/data/test.pkl'
train_path = 'D-tasks/data/train.pkl'
meta_path = 'D-tasks/data/meta.pkl'

class GrainedClassifierCNN(nn.Module):
    def __init__(
        self,
        fine_classes: int = 100,
        coarse_classes: int = 20,
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

        # Individual output layers for fine and coarse classes
        self.fine = nn.Linear(hidden_channels, fine_classes)
        self.coarse = nn.Linear(hidden_channels, coarse_classes)

    def forward(self, x):
        z = self.f(x)  # (B, feat_dim)

        z = self.proj(z) # (B, C*H*W)
        z = z.view(z.size(0), self.reshape_channels,
                   self.fmap_size, self.fmap_size) # (B, C, H, W)

        z = self.h(z) # (B, C, 1, 1)
        z = z.view(z.size(0), -1) # (B, C)

        fine_logits = self.fine(z)
        coarse_logits = self.coarse(z)
        return fine_logits , coarse_logits

def accuracy_top1(logits, targets):
    """
    Computes top-1 accuracy.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
    return correct / total

def mixup_data(x, y_fine, y_coarse, alpha=0.2):
    """
    Data Augmentation method mix up
    Overlaying 2 images into 1 image
    """
    if alpha <= 0:
        return x, y_fine, y_fine, y_coarse, y_coarse, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_f_a, y_f_b = y_fine, y_fine[index]
    y_c_a, y_c_b = y_coarse, y_coarse[index]

    return mixed_x, y_f_a, y_f_b, y_c_a, y_c_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, dataloader, optimizer, criterion, device, weight_fine=1, weight_coarse=1):
    model.train()
    running_loss = 0.0
    running_acc_fine = 0.0
    running_acc_coarse = 0.0
    running_acc_exact = 0.0
    total_samples = 0

    for images, fine_labels, coarse_labels in dataloader:
        images = images.to(device)
        fine_labels = fine_labels.to(device)
        coarse_labels = coarse_labels.to(device)
        
        # Forward
        mixed_images, y_f_a, y_f_b, y_c_a, y_c_b, lam = mixup_data(images, fine_labels, coarse_labels, 0.1)
        fine_logits, coarse_logits = model(mixed_images)

        fine_loss = mixup_criterion(criterion, fine_logits, y_f_a, y_f_b, lam)
        coarse_loss = mixup_criterion(criterion, coarse_logits, y_c_a, y_c_b, lam)

        loss = weight_fine * fine_loss+ weight_coarse * coarse_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc_fine += accuracy_top1(fine_logits, fine_labels) * batch_size
        running_acc_coarse += accuracy_top1(coarse_logits, coarse_labels) * batch_size

        pred_f = fine_logits.argmax(dim=1)
        pred_c = coarse_logits.argmax(dim=1)
        running_acc_exact += ((pred_f == fine_labels) & (pred_c == coarse_labels)).sum().item()

        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc_fine = running_acc_fine / total_samples
    epoch_acc_coarse = running_acc_coarse / total_samples
    epoch_acc_exact = running_acc_exact / total_samples
    return epoch_loss, epoch_acc_fine, epoch_acc_coarse, epoch_acc_exact

def evaluate(model, dataloader, criterion, device,  weight_fine=1, weight_coarse=1):
    model.eval()
    running_loss = 0.0
    running_acc_fine = 0.0
    running_acc_coarse = 0.0
    running_acc_exact = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, fine_labels,coarse_labels in dataloader:
            images = images.to(device)
            fine_labels = fine_labels.to(device)
            coarse_labels = coarse_labels.to(device)

            fine_logits, coarse_logits = model(images)
            loss_fine = criterion(fine_logits, fine_labels)
            loss_coarse = criterion(coarse_logits, coarse_labels)
            loss = weight_fine * loss_fine + weight_coarse * loss_coarse

            batch_size = images.size(0)

            total_samples += batch_size
            running_loss += loss.item() * batch_size
            fine_pred = fine_logits.argmax(dim=1)
            coarse_pred = coarse_logits.argmax(dim=1)
            running_acc_fine += (fine_pred == fine_labels).sum().item()
            running_acc_coarse += (coarse_pred == coarse_labels).sum().item()
            running_acc_exact += ((fine_pred == fine_labels) &
                                  (coarse_pred == coarse_labels)).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc_fine = running_acc_fine / total_samples
    epoch_acc_coarse = running_acc_coarse / total_samples
    epoch_acc_exact = running_acc_exact / total_samples

    return epoch_loss, epoch_acc_fine, epoch_acc_coarse, epoch_acc_exact

def main():
    # Hyperparameters
    num_epochs = 250
    batch_size = 16
    lr_backbone = 5e-4
    lr_head = 2e-3
    weight_decay = 5e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_dataset, test_dataset = dataset.create_dataset(train_path, test_path,meta_path) #to change

    train_loader = DataLoader(
        dataset =train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset= test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = GrainedClassifierCNN()
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

    train_losses, train_accs_fine, train_accs_coarse, train_accs_exact = [], [], [],[]
    test_losses, test_accs_fine, test_accs_coarse, test_accs_exact = [], [],[],[]
    weight_fine, weight_coarse = 1 ,1 # Weightings for combined accuracy

    best_test_acc = 0.0

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss,train_acc_fine, train_acc_coarse, train_acc_exact= train_one_epoch(
            model, train_loader, optimizer, criterion, device, weight_fine, weight_coarse
        )
        test_loss,test_acc_fine, test_acc_coarse, test_acc_exact = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        train_losses.append(train_loss)
        train_accs_fine.append(train_acc_fine)
        train_accs_coarse.append(train_acc_coarse)
        train_accs_exact.append(train_acc_exact)
        
        test_losses.append(test_loss)
        test_accs_fine.append(test_acc_fine)
        test_accs_coarse.append(test_acc_coarse)
        test_accs_exact.append(test_acc_exact)

        # Save best model based on test accuracy
        if test_acc_exact > best_test_acc:
            best_test_acc = test_acc_exact
            torch.save(model.state_dict(), "D-tasks\models\d3.pth")

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] \n "
            f"Train Loss: {train_loss:.4f} | Fine Train Acc: {train_acc_fine*100:.2f}% | Coarse Train Acc: {train_acc_coarse*100:.2f}% | | Combined Train Acc: {train_acc_exact*100:.2f}% | \n"
            f"Test Loss: {test_loss:.4f} | Fine Test Acc: {test_acc_fine*100:.2f}% | Coarse Test Acc: {test_acc_coarse*100:.2f}% | | Combined Test Acc: {test_acc_exact*100:.2f}% | \n"
            
        )

    # Store training logs
    run_id = RUN_ID
    log_path = f"D-tasks/logs/training_{run_id}.npz"

    np.savez(
        log_path,
        train_losses=train_losses,
        train_accs_coarse = train_accs_coarse,
        train_accs_fine = train_accs_fine,
        train_accs=train_accs_exact,
        test_accs_coarse = test_accs_coarse,
        test_accs_fine = test_accs_fine,
        test_losses=test_losses,
        test_accs=test_accs_exact,
    )

    print(f"Best Test Acc: {best_test_acc*100:.2f}%")


def prepare_test():
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output two tensors: the first of shape (B, 100), with the second of shape
    #  (B, 20). B is the batch size and 100/20 is the number of fine/coarse classes.
    #  The output is the prediction of your classifier, providing two scores for both fine and coarse classes,
    #  for each image in input

    model = GrainedClassifierCNN()

    # do not edit from here downwards
    weights_path = 'D-tasks\models\d3.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model

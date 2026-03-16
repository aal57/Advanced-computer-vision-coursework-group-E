import torch
from torch import nn
import torch.nn.functional as F
from backbone import new_backbone
import torch.optim as optim
from dataset import ZeroshotDataset
from torch.utils.data import DataLoader
from task_d1 import FineGrainedClassifierCNN, train_one_epoch, evaluate
# TODO save your best model and store it at './models/d5.pth'

test_path = 'D-tasks/data/zero_shot/test.pkl'
train_path = 'D-tasks/data/zero_shot/train.pkl'
support_1_path = 'D-tasks/data/zero_shot/support_1.pkl'
support_5_path = 'D-tasks/data/zero_shot/support_5.pkl'
support_10_path = 'D-tasks/data/zero_shot/support_10.pkl'

# knn classifier with softmax
def knn_predict_weighted_softmax(query_Z, support_Z, support_Y, k, temperature=0.07, metric = "cosine"):
    # Metrics
    if metric == "cosine":
        sims = query_Z @ support_Z.T
    if metric == "euclidian":
        sims = -torch.cdist(query_Z, support_Z, p=2)

    topk_sims, topk_idx = torch.topk(sims, k=k, dim=1)
    topk_labels = support_Y[topk_idx]

    weights = torch.softmax(topk_sims / temperature, dim=1)

    num_classes = int(support_Y.max().item()) + 1
    preds = []

    for i in range(topk_labels.size(0)):
        class_scores = torch.zeros(
            num_classes, device=weights.device
        )
        for j in range(k):
            cls = topk_labels[i, j]
            class_scores[cls] += weights[i, j]
        preds.append(class_scores.argmax().item())

    return torch.tensor(preds, dtype=torch.long)

# Transforms image by flipping horizontally
def tta_transforms(img):
    crops = []
    crops.append(img)

    crops.append(torch.flip(img, dims=[2]))

    return crops

# Extract embeddings
def extract_embeddings_tta(backbone, dataloader, device):
    backbone.eval()
    all_z = []
    all_y = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels, _ = batch

            images = images.to(device)

            batch_embeds = []

            for i in range(images.size(0)):
                img = images[i]
                views = tta_transforms(img)

                z_views = []
                for v in views:
                    v = v.unsqueeze(0)           
                    z = backbone(v)              
                    z_views.append(z)

                z_mean = torch.mean(
                    torch.cat(z_views, dim=0),
                    dim=0
                )
                batch_embeds.append(z_mean)

            Z = torch.stack(batch_embeds, dim=0)
            Z = F.normalize(Z, dim=1)

            all_z.append(Z.cpu())
            all_y.append(labels.cpu())

    return torch.cat(all_z), torch.cat(all_y)

def evaluate_knn(backbone, support_loader, query_loader, device, ks=(1,3,5,7,9,11), metric = "cosine"):
    support_Z, support_Y = extract_embeddings_tta(backbone, support_loader, device)
    query_Z, query_Y = extract_embeddings_tta(backbone, query_loader, device)

    unique = torch.unique(support_Y)

    mapping = {int(lbl): i for i, lbl in enumerate(unique.tolist())}
    support_Y_mapped = torch.tensor([mapping[int(y)] for y in support_Y.tolist()], dtype=torch.long)
    query_Y_mapped = torch.tensor([mapping.get(int(y), -1) for y in query_Y.tolist()], dtype=torch.long)

    # Filter out query samples where label not in support mapping 
    valid = query_Y_mapped >= 0
    query_Z = query_Z[valid]
    query_Y_mapped = query_Y_mapped[valid]

    results = {}
    for k in ks:
        preds = knn_predict_weighted_softmax(
            query_Z,
            support_Z,
            support_Y_mapped,
            k=k,
            temperature=0.07,
            metric = metric
        )
        acc = (preds == query_Y_mapped).float().mean().item()
        results[k] = acc
    return results

def evaluate_multiple_support_sets(backbone, support_loaders, query_loader, device, ks=(1, 3, 5, 7, 9, 11),metric = "cosine"):

    all_results = {}

    for support_name, support_loader in support_loaders.items():
        results = evaluate_knn(
            backbone=backbone,
            support_loader=support_loader,
            query_loader=query_loader,
            device=device,
            ks=ks,
            metric = metric
        )
        all_results[support_name] = results

    return all_results

def print_results_table(all_results):
    print("/nZero-shot k-NN results")

    for support_name, result_dict in all_results.items():
        print(f"{support_name}:")
        for k, acc in result_dict.items():
            print(f"  k={k:<2d} -> acc={acc*100:.2f}%")
        print()

# Training new model (if needed)
def training(device,train_loader, test_loader):
    model = FineGrainedClassifierCNN()
    model.to(device)

    num_epochs = 200
    lr_backbone = 5e-4
    lr_head = 1e-3
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        [
            {"params": model.f.parameters(), "lr": lr_backbone},
            {"params": model.proj.parameters(), "lr": lr_head},
            {"params": model.h.parameters(), "lr": lr_head},
            {"params": model.fc.parameters(), "lr": lr_head},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.97),
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[130, 180],
        gamma=0.15
    )

    best_train_acc = 0.0

    print("/nTraining FineGrainedClassifierCNN on zero-shot.../n")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step()

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), "D-tasks/models/d5.pth")

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )


    print(f"/nBest classifier train accuracy on zero-shot: {best_train_acc*100:.2f}%")

def main():
    # Create zero-shot split
    train_dataset = ZeroshotDataset(train_path)
    test_dataset = ZeroshotDataset(test_path)
    support_1_dataset = ZeroshotDataset(support_1_path)
    support_5_dataset = ZeroshotDataset(support_5_path)
    support_10_dataset = ZeroshotDataset(support_10_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders for training and testing
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # Comment out training if model has already been trained
    training(device, train_loader, test_loader)
    
    # Load best trained model
    best_model = FineGrainedClassifierCNN()
    state_dict = torch.load("D-tasks/models/d5.pth", map_location=device)
    best_model.load_state_dict(state_dict)
    best_model.to(device)
    best_model.eval()

    backbone = best_model.f
    backbone.to(device)
    backbone.eval()

    # Create support and query loaders
    support_loaders = {
        "1-shot": DataLoader(support_1_dataset, batch_size=16, shuffle=False, num_workers=0),
        "5-shot": DataLoader(support_5_dataset, batch_size=16, shuffle=False, num_workers=0),
        "10-shot": DataLoader(support_10_dataset, batch_size=16, shuffle=False, num_workers=0),
    }

    query_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0
    )

    # Evaluate with k-NN
    metric = "cosine"
    all_results = evaluate_multiple_support_sets(
        backbone=backbone,
        support_loaders=support_loaders,
        query_loader=query_loader,
        device=device,
        ks=(1, 3, 5, 7, 9, 11),
        metric = metric
    )

    print_results_table(all_results)

#main()

def prepare_test():
    # TODO: Load the model and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply return model.backbone

    model = FineGrainedClassifierCNN()

    # do not edit from here downwards
    weights_path = 'D-tasks\models\d5.pth'
    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model.f
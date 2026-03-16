import task_d1
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
from task_d1 import FineGrainedClassifierCNN   

train_path = 'D-tasks/data/train.pkl'
test_path = 'D-tasks/data/test.pkl'
meta_path = 'D-tasks/data/meta.pkl'
d1_path = "D-tasks/models/d1.pth"

def extract_embeddings_coarse(backbone, dataloader, device, normalize=True):
    backbone.eval()
    all_z = []
    all_y = []

    for batch in dataloader:
        images, _, coarse_labels = batch

        images = images.to(device)
        z = backbone(images)  

        if normalize:
            z = F.normalize(z, p=2, dim=1)

        all_z.append(z.cpu())
        all_y.append(coarse_labels.cpu())

    Z = torch.cat(all_z, dim=0)  
    Y = torch.cat(all_y, dim=0)  
    return Z, Y

# knn classifier with softmax
def knn_predict_weighted_softmax(query_Z, support_Z, support_Y, k, temperature=0.07,metric="cosine"):
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

def evaluate_knn_coarse(backbone, train_loader, test_loader, device, ks=(1, 3, 5, 10), normalize=True,metric="cosine"):
    train_Z, train_Y = extract_embeddings_coarse(backbone, train_loader, device, normalize)
    test_Z, test_Y = extract_embeddings_coarse(backbone, test_loader, device, normalize)

    results = {}

    for k in ks:
        preds = knn_predict_weighted_softmax(test_Z, train_Z, train_Y, k=k,temperature=0.7,metric=metric)
        acc = (preds == test_Y).float().mean().item()
        results[k] = acc

    return results

def print_knn_results(results):
    print("/nCoarse-label k-NN results from D1 backbone")
    for k, acc in results.items():
        print(f"k={k:<2d} -> acc={acc*100:.2f}%")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset = dataset.create_dataset(train_path, test_path, meta_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0
    )

    # Load D1 fine-grained model
    model = FineGrainedClassifierCNN()
    state_dict = torch.load(d1_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Use only backbone
    backbone = model.f
    backbone.to(device)
    backbone.eval()

    # Evaluate with k-NN
    metric = "euclidian"
    results = evaluate_knn_coarse(
        backbone=backbone,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        ks=(1, 3, 5, 7, 9,11),
        normalize=True,
        metric= metric
    )

    print_knn_results(results)

def prepare_test():
    # TODO: Load the model from task D1 and return its **backbone**. The backbone model will be fed a batch of images,
    #  i.e. a tensor of shape (B, 3, 32, 32), where B >= 2, and must return a tensor of shape (B, 576), i.e.
    #  the embedding extracted for the input images. Hint: if the backbone is stored inside your model with the
    #  name "backbone", you can simply leave the code below as is. Otherwise, please adjust.

    model = task_d1.prepare_test()
    return model.f

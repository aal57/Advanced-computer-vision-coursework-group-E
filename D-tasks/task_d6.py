from sklearn.manifold import TSNE
import torch
import dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from backbone import new_backbone
import torch.nn.functional as F
from torch import nn
import numpy as np

#Architecture for our coarse grain D2 classifier
class CoarseGrainedClassifierCNN(nn.Module):
    """
    Full model = backbone f + CNN classifier h.
    f stays EXACTLY as provided by new_backbone().
    """
    def __init__(
        self,
        num_classes: int = 20,
        hidden_channels: int = 512,
        dropout: float = 0.25,
    ):
        super().__init__()

        # ----- Backbone f (unchanged) -----
        self.f = new_backbone()  # returns embedding vector for each image

        # ----- Infer embedding dimension (unchanged) -----
        with torch.no_grad():
            self.f.eval()
            dummy = torch.zeros(1, 3, 32, 32)
            feat_dim = self.f(dummy).shape[1]     # e.g., 576 or 960
        self.f.train()

        # ----- CNN classifier head h -----
        # Convert vector → (C,H,W), apply small CNN, then flatten → FC → logits

        # Choose a small square feature map size for CNN, e.g., 4×4
        fmap_size = 6
        self.fmap_size = fmap_size

        self.reshape_channels = hidden_channels

        self.proj = nn.Linear(feat_dim, hidden_channels * fmap_size * fmap_size)

        self.h = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # → (B, hidden_channels, 1, 1)
        )

        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        # ----- Backbone -----
        z = self.f(x)  # shape (B, feat_dim)

        # ----- Project + reshape -----
        z = self.proj(z)                                     # (B, C*H*W)
        z = z.view(z.size(0), self.reshape_channels,
                   self.fmap_size, self.fmap_size)           # (B, C, H, W)

        # ----- CNN Head -----
        z = self.h(z)                                        # (B, C, 1, 1)
        z = z.view(z.size(0), -1)                            # (B, C)

        # ----- Final classifier -----
        logits = self.fc(z)
        return logits


tsne = TSNE(random_state=1, n_components=2, max_iter=5000, metric="euclidean")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(filepath):
    model = new_backbone().to(device) # define architecture
    state_dict = torch.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model.eval()

coarse_d4 = load_model(f"D-tasks/models/d4_m=0.8updated_coarse.pth")
    
def get_embeddings_and_labels(model, fine=False):
    
    test_path = 'D-tasks/data/test.pkl'
    train_path = 'D-tasks/data/train.pkl'
    meta_path = 'D-tasks/data/meta.pkl'
    test_dataset, train_dataset = dataset.create_dataset(train_path, test_path, meta_path)

    #loads datasets and prepares them for training
    train_loader = DataLoader(
        dataset = train_dataset, batch_size=500)
    test_loader = DataLoader(
        dataset= test_dataset, batch_size=500)
    
    embeddings = []
    labels = []
    
    for images, fine_labels, coarse_labels in test_loader:
        images = images.to(device)
        with torch.no_grad():
            batch_embeddings = model(images)
            embeddings.append(batch_embeddings.cpu())
            
            if fine:
                labels+=fine_labels.tolist()
            else:
                labels+=coarse_labels.tolist()
    
    embeddings = torch.cat(embeddings)
    return F.normalize(embeddings).numpy(), labels

embeddings, labels = get_embeddings_and_labels(coarse_d4, fine=False)

fine_metric = tsne.fit_transform(embeddings)

unique_labels = np.unique(labels)
legend_labels = []
for label in unique_labels:
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=str(label), markerfacecolor=plt.cm.tab20(label / max(unique_labels)), markersize=8))


plt.figure(figsize=(10,10))
plt.scatter(fine_metric[:,0], fine_metric[:,1], c=labels, cmap="tab20")
plt.xlabel("dim1")
plt.ylabel("dim2")
#plt.title("D4 Embeddings, m=0.5 - Metric Learning")
plt.title("D4 Embeddings, semi-hard sampling, m=0.8")
plt.legend(handles=legend_labels, title="Classes", ncol=2, loc="lower left")
plt.show()


test_path = 'D-tasks/data/test.pkl'
train_path = 'D-tasks/data/train.pkl'
meta_path = 'D-tasks/data/meta.pkl'
test_dataset, train_dataset = dataset.create_dataset(train_path, test_path, meta_path)
    
test_loader = DataLoader(
        dataset= test_dataset, batch_size=500)

coarse_d2 = CoarseGrainedClassifierCNN()
coarse_d2.to(device) #define architecture
state_dict = torch.load("D-tasks/models/d2.pth", map_location=device)
coarse_d2.load_state_dict(state_dict)
coarse_d2.to(device)


embeddings = []
labels = []

with torch.no_grad():
    for images, _, coarse_labels in test_loader:
        images = images.to(device)
        
        inferred = coarse_d2.f(images)
        
        inferred = coarse_d2.proj(inferred)
        inferred = inferred.view(inferred.size(0), coarse_d2.reshape_channels, coarse_d2.fmap_size, coarse_d2.fmap_size)
        
        inferred = coarse_d2.h(inferred)
        inferred = inferred.view(inferred.size(0),-1)
        
        embeddings.append(inferred.cpu())
        labels.append(coarse_labels.cpu())
        
embeddings = torch.cat(embeddings)
labels = torch.cat(labels)

fine_metric = tsne.fit_transform(embeddings)

unique_labels = np.unique(labels)
legend_labels = []
for label in unique_labels:
    legend_labels.append(plt.Line2D([0], [0], marker='o', color='w', label=str(label), markerfacecolor=plt.cm.tab20(label / max(unique_labels)), markersize=8))


plt.figure(figsize=(10,10))
plt.scatter(fine_metric[:,0], fine_metric[:,1], c=labels, cmap="tab20")
plt.xlabel("dim1")
plt.ylabel("dim2")
plt.title("D2 Embeddings - Decision Boundary Learning")
plt.legend(handles=legend_labels, title="Classes", ncol=2, loc="lower right")
plt.show()
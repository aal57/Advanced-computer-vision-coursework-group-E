from collections import Counter
import torch
import pickle
from backbone import new_backbone
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import dataset
import torch.optim as optim
from torch.utils.data import Sampler
import numpy as np
from torchvision import transforms

import csv

#function for calculating distance metrics set by default to l2 normalised euclidean distances.
def calculate_distances(embeddings, metric="euclidean"):
    if metric=="euclidean":
        embeddings = F.normalize(embeddings)
        return torch.cdist(embeddings, embeddings)
    elif metric=="cosine":
        embeddings = F.normalize(embeddings)
        return 1 - torch.matmul(embeddings, embeddings.t())
    elif metric=="l1":
        embeddings = F.normalize(embeddings, p=1)
        return torch.cdist(embeddings, embeddings)

# TODO Save your best models and store them at './models/d4_m={margin}_fine.pth' or ./models/d4_m={margin}_coarse.pth,
#  depending on whether you trained the model with triplets formed with the fine or coarse labels.
#  {margin} is the margin value that you used to train the model. You must upload at least two models, one for the
#  fine-grained version and one for the coarse-grained version, specifying the margin value. You can upload multiple
#  models trained with different margin values

#Sampler to ensure positive and negative examples for each anchor in a given batch
class CustomSampler(Sampler):
    #batch size = total number of samples per batch, labels per batch = classes in each batch, and each class gets batch size/labels per batch number of samples
    def __init__(self, labels, batch_size=64, labels_per_batch=16):
        self.labels = labels
        #gets all unique labels in labels
        self.unique_labels = list(set(labels.tolist()))
        
        self.batch_size = batch_size
        #calculates the number of batches per epoch, used in __len__
        self.num_batches = len(self.labels) //self.batch_size
        
        #ensures the number of labels per batch is valid
        self.labels_per_batch = labels_per_batch
        if batch_size%labels_per_batch!=0:
            raise ValueError("Batch size must be divisible by labels per batch")
        
        #gets number of samples for each class. Equal for all classes.
        self.samples_per_label = batch_size // self.labels_per_batch
        self.indices_by_labels = {label: np.where(np.array(labels) == label)[0] for label in self.unique_labels}
        
    def __iter__(self):
        
        # get random sample of all labels
        for i in range(self.num_batches):
            batch = []
            
            #selects a subset of labels for the given batch
            batch_labels = np.random.choice(self.unique_labels, self.labels_per_batch, replace=False)
            
            
            for label in batch_labels:
                indices = self.indices_by_labels[label]

                selected_indices = np.random.choice(indices, self.samples_per_label, replace=len(indices) < self.samples_per_label)

                batch.extend(selected_indices)
            
            np.random.shuffle(batch)
            yield from batch
            
    
    def __len__(self):
        return self.batch_size * self.num_batches

#calculates the distance for anchors-positives in a set of embeddings
def calc_distance_ap(embeddings, labels):
    distances = calculate_distances(embeddings)
    labels = labels.unsqueeze(1)
    
    positive_mask = labels.eq(labels.t())
    positive_mask.fill_diagonal_(False)
    
    positive_dists = distances[positive_mask]
    return positive_dists.mean().item()

#calculates distances for anchors-negatives in a set of embeddings
def calc_distance_an(embeddings, labels):
    distances = calculate_distances(embeddings)
    labels = labels.unsqueeze(1)
    
    negatives_mask = ~labels.eq(labels.t())
    negatives_mask.fill_diagonal_(False)
    
    negative_dists = distances[negatives_mask]
    return negative_dists.mean().item()

#hard triplet loss
def hard_triplet_loss(embeddings, labels, margin):
    distances = calculate_distances(embeddings)
    labels = labels.unsqueeze(1)
    
    #creates a mask for all values of the same type as a given label.
    positive_mask = labels.eq(labels.t())
    #ensures the same value isn't considered valid for triplet positive
    positive_mask.fill_diagonal_(False)
    #masked fill avoids -inf values.
    hardest_pos = distances.masked_fill(~positive_mask, -1e9).max(dim=1)[0]
    
    #creates a mask, setting all valid negatives to true
    negatives_mask = ~positive_mask
    negatives_mask.fill_diagonal_(False)
    hardest_neg = distances.masked_fill(~negatives_mask, float("inf")).min(dim=1)[0]
    
    loss = torch.relu(hardest_pos-hardest_neg+margin)
    
    return loss.mean(), len(embeddings)

def semi_hard_triplet_loss(embeddings, labels, margin):
    distances = calculate_distances(embeddings)
    labels = labels.unsqueeze(1)
    
    #creates a mask for all values of the same type as a given label.
    positive_mask = labels.eq(labels.t())
    #ensures the same value isn't considered valid for triplet
    positive_mask.fill_diagonal_(False)
    hardest_pos = distances.masked_fill(~positive_mask, -1e9).max(dim=1)[0]
    
    #creates a mask, setting all valid negatives to true (in contrast to hard requires triplets to be semi-hard)
    semi_hard_neg_mask = (~positive_mask) & (distances>hardest_pos.unsqueeze(1)) & (distances< hardest_pos.unsqueeze(1)+margin)
    semi_hard_neg_mask.fill_diagonal_(False)
    semi_hard_neg = distances.masked_fill(~semi_hard_neg_mask, float("inf")).min(dim=1)[0]
    
    loss = torch.relu(hardest_pos-semi_hard_neg+margin)
    
    return loss.mean(), 0

#hard if no semi-hard triplet loss
def hard_if_no_semi_hard_triplet_loss(embeddings, labels, margin):
    distances = calculate_distances(embeddings)
    labels = labels.unsqueeze(1)
    
    positive_mask = labels.eq(labels.t())
    positive_mask.fill_diagonal_(False)
    hardest_pos = distances.masked_fill(~positive_mask, -1e9).max(dim=1)[0]
    
    semi_hard_neg_mask = (~positive_mask) & (distances>hardest_pos.unsqueeze(1)) & (distances< hardest_pos.unsqueeze(1)+margin)
    semi_hard_neg_mask.fill_diagonal_(False)
    semi_hard_neg = distances.masked_fill(~semi_hard_neg_mask, float("inf")).min(dim=1)[0]
    
    negative_mask = ~positive_mask
    negative_mask.fill_diagonal_(False)
    
    hard_neg = distances.masked_fill(~negative_mask, float("inf")).min(dim=1)[0]
    
    #checks for -inf gradients contributing nothing, creates a mask then picks hard negatives for true values
    use_hard = torch.isinf(semi_hard_neg)
    chosen_neg = torch.where(use_hard, hard_neg, semi_hard_neg)
    
    #counts num of semi-hard and hard negatives.
    semi_hard_negatives_count = (~use_hard).sum().item()
    hard_negatives_count = use_hard.sum().item()
    
    #avoids divide by 0
    if semi_hard_negatives_count>0:
        hard_neg_percentage = (hard_negatives_count/semi_hard_negatives_count) * 100
    else:
        hard_neg_percentage = 0
    
    loss = torch.relu(hardest_pos-chosen_neg+margin)
    
    return loss.mean(), hard_neg_percentage
    
def run_epoch(epoch, total_epochs, model, loader, optimizer, device, margin = 0.2, fine=True, test=False, epoch_record=None):

    #artifact of other sampling method since dropped.
    progress = (epoch /total_epochs)
    
    #ensures model set to correct mode.
    if not test:
        model.train()
    else:
        model.eval()
        
    total_loss = 0.0
    
    mean_hard_neg_percentage = 0.0
    
    #iterates through all images, runs batch loss for each batch
    for images, fine_labels, coarse_labels in loader:
        
        images = images.to(device)
        if fine:
            labels = fine_labels.to(device)
        else:
            labels = coarse_labels.to(device)
        
        if not test:
            embeddings = model(images)

            loss, batch_hard_neg_percentage =  semi_hard_triplet_loss(embeddings, labels, margin=margin)
            mean_hard_neg_percentage+=batch_hard_neg_percentage
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                embeddings = model(images)
                loss, _ = semi_hard_triplet_loss(embeddings, labels, margin=margin)

                
                        
        total_loss += loss.item()
        
    return total_loss / len(loader), mean_hard_neg_percentage / len(loader)


#recall at k calculation.
def recall_at_k(embeddings, labels, k=1): 
    class_sizes = torch.bincount(labels)
    
    distances = calculate_distances(embeddings)
    distances.fill_diagonal_(float('inf'))
    
    #gets smallest distances
    knns = distances.topk(k, largest=False).indices
    #gets labels of closest neighbours
    knn_labels = labels[knns]
    
    correct = knn_labels.eq(labels.unsqueeze(1))
    
    recall_per_sample = correct.sum(dim=1) / (class_sizes[labels])
    return recall_per_sample.mean().item()


#calculates recall at all k values, alongside mean AN and AP distances. 
def calculate_performance(k_values=[5,10,50,100], margin=0.2, use_fine_labels=False, model=None, test_loader=None):
    """
    gets recall and mean AN and AP results for a given model iteration

    Args:
        k_values (list, optional): k values to get recall@k for. Defaults to [5,10,50,100].
        margin (float, optional): the margin of the model. Defaults to 0.2.
        use_fine_labels (bool, optional): whether the model uses fine labels. Defaults to False.
        model (MobileNetV3, optional): the model to use, can be none, if so will load an existing model. Defaults to None.
        test_loader (DataLoader, optional): pre-existing loader for test data. Defaults to None.

    Returns:
        tuple: recall@5, then a list containing recall@k values specified, followed by mean AP and mean AN distance.
    """
    
    if test_loader is None:
        # paths to data
        test_path = 'D-tasks/data/test.pkl'
        train_path = 'D-tasks/data/train.pkl'
        meta_path = 'D-tasks/data/meta.pkl'
        test_dataset, _ = dataset.create_dataset(train_path, test_path, meta_path) #to change
        
        test_loader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #below function to run without retraining model.
    if model is None:
        model = new_backbone().to(device)

        if use_fine_labels:
            state_dict = torch.load(f"D-tasks/models/d4_m={margin}_fine.pth", map_location=device)
        else:
            state_dict = torch.load(f"D-tasks/models/d4_m={margin}_coarse.pth", map_location=device)

        model.load_state_dict(state_dict)
    
    model.eval()
    
    #loads all embeddings and labels.
    embeddings = []
    labels = []
    for images, fine_labels, coarse_labels in test_loader:
        images = images.to(device)
        with torch.no_grad():
            batch_embeddings = model(images)
            embeddings+=batch_embeddings.cpu()
            if use_fine_labels:
                labels+=fine_labels.tolist()
            else:
                labels+=coarse_labels.tolist()
    
    labels = torch.tensor(labels)
    embeddings = torch.stack(embeddings)
    
    stats = []
    
    recall_at_5 = 0 
    
    # loops through k values, getting recall at k for each and printing to terminal
    # special addition hard coded for recall@5 for use in early stopping
    for k in k_values:
        recall = recall_at_k(embeddings, labels, k=k)
        if k == 5:
            recall_at_5 = recall 
        stats.append(recall)
        print(f"Recall@{k}: {recall:.4f}")
    
    mean_ap_dist = calc_distance_ap(embeddings, labels)
    stats.append(mean_ap_dist)
    print(f"Mean AP distance: {mean_ap_dist}")
    mean_an_dist = calc_distance_an(embeddings, labels)
    stats.append(mean_an_dist)
    print(f"Mean AN distance: {mean_an_dist}")
    
    return recall_at_5, stats
    
    
        
        
def train_model(margin=0.2, num_epochs = 250, lr = 2e-3, weight_decay = 1e-4, batch_size=256, eta_min=1e-5,use_fine_labels=False, epoch_record=None, model_name=""):
    """
    function to train a new model

    Args:
        margin (float, optional): margin for the model to use. Defaults to 0.2.
        num_epochs (int, optional): number of epochs (assuming no early stopping). Defaults to 250.
        lr (float, optional): learning rate. Defaults to 2e-3.
        weight_decay (float, optional): weight decay value. Defaults to 1e-4.
        batch_size (int, optional): batch size. Defaults to 256.
        eta_min (float, optional): final learning rate, assuming no early stopping. Defaults to 1e-5.
        use_fine_labels (bool, optional): whether to use fine labels. Defaults to False.
        epoch_record (str, optional): file to save results to. Defaults to None.
        model_name (str, optional): name to save the model to, actually saves to d4_m=MODEL_NAME. Defaults to "".
    """
    #Just prints which labels are being used
    if use_fine_labels:
        print("Training with fine labels")
    else:
        print("Training with coarse labels")
    
    # paths to data
    test_path = 'D-tasks/data/test.pkl'
    train_path = 'D-tasks/data/train.pkl'
    meta_path = 'D-tasks/data/meta.pkl'    

    #ensures the gpu is used if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  Datasets and loaders
    train_dataset, test_dataset = dataset.create_dataset(train_path, test_path,meta_path) #to change
    
    #creates a sampler to ensure valid samples collected for triplet loss
    if use_fine_labels:
        sampler = CustomSampler(train_dataset.fine_labels, batch_size=batch_size)
    else:
        sampler = CustomSampler(train_dataset.coarse_labels, batch_size=batch_size)
    
    #loads datasets and prepares them for training
    train_loader = DataLoader(
        dataset = train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(
        dataset= test_dataset, batch_size=batch_size, shuffle=False)

    #sets up the model and optimiser
    model = new_backbone().to(device)

    #sets up optimiser and scheduler for learning.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)  
    
    #adds data for 0th epoch to csv
    _, stats = calculate_performance(k_values=[5,10,50,100],margin=margin, use_fine_labels=use_fine_labels, model=model, test_loader=test_loader)
    epoch_record.writerow([model_name, 0, margin, "fine" if use_fine_labels else "coarse", *stats]+[1,1,0])
    
    
    highest_recall_at_5 = 0
    since_improved = 0
        
    # basic training loop
    for epoch in range(1, num_epochs + 1):
        #only uses hard triplet mining every 4 epochs
        loss, mean_hard_neg_percentage = run_epoch(epoch, num_epochs,model, train_loader, optimizer, device, margin=margin, fine=use_fine_labels)
        
        #steps the learning rate scheduler
        scheduler.step()
        if epoch % 5 == 0:
            #stops training if recall hasn't improved in 20 epochs
            if since_improved >5:
                return
            
            #gives training lost and test loss each epoch (for me to see if overfitting)
            print(f"Epoch [{epoch:03d}]")
            print("Train loss: {:.4f}".format(loss))
            test_loss, _ = run_epoch(epoch, num_epochs, model, test_loader, optimizer, device, margin=margin, fine=use_fine_labels, test=True)
            print("Test Loss: {:.4f}".format(test_loss))
            recall_at_5, stats = calculate_performance(k_values=[5,10,50,100],margin=margin, use_fine_labels=use_fine_labels, model=model, test_loader=test_loader)
            
            #    epoch_record.writerow(['technique', 'epoch', 'margin', 'fine_labels', 'recall_at_5','recall_at_10','recall_at_50','recall_at_100','mean_AP_distance','mean_AN_distance','train_loss','test_loss'])

            epoch_record.writerow([model_name, epoch, margin, "fine" if use_fine_labels else "coarse", *stats, loss, test_loss, mean_hard_neg_percentage])
            
            if recall_at_5 > highest_recall_at_5:
                highest_recall_at_5 = recall_at_5
                since_improved = 0
                if use_fine_labels:
                    print("Saving model for fine labels")
                    torch.save(model.state_dict(), f"D-tasks/models/{model_name}.pth")
                else:
                    print("Saving model for coarse labels")
                    torch.save(model.state_dict(), f"D-tasks/models/d4_m={model_name}_coarse.pth")
            else:
                since_improved += 1
    
    #acknowledgement of succesful model saving
    print("Model Saved!")
    

def main():
    with open('D-tasks/batch_sizes.csv', 'a', newline='') as f:
        epoch_record = csv.writer(f)
        # epoch_record.writerow(['technique', 'epoch', 'margin', 'fine_labels', 'recall_at_5','recall_at_10','recall_at_50','recall_at_100','mean_AP_distance','mean_AN_distance','train_loss','test_loss', 'mean_hard_negative_percentage'])
        for margin in [0.5]:
            for batch_size in [64]:
                #best train values!!
                #train_model(margin=margin, lr=1e-3, weight_decay=1e-5, batch_size=128, num_epochs=250, eta_min=5e-7, use_fine_labels=use_fine_labels)
                train_model(margin=margin, lr=1e-3, weight_decay=1e-5, batch_size=batch_size, num_epochs=250, eta_min=5e-7, use_fine_labels=False, epoch_record=epoch_record, model_name=str(margin)+"updated")
     

def prepare_test(margin, fine_labels):
    # TODO: Create an instance of your model here. Your model must take in input a tensor of shape
    #  (B, 3, 32, 32), where B >= 2, and output a tensor of shape (B, 576), where B is the batch size and 576 is the
    #  embedding dimension. Make sure that the correct model is loaded depending on the margin and fine_labels parameters
    #  where `margin` is a float and `fine_labels` is a boolean that if True/False will load the model trained with triplets
    #  formed with the fine/coarse labels.
    
    model = new_backbone() # TODO change this to your model

    # do not edit from here downwards
    s = 'fine' if fine_labels else 'coarse'
    weights_path = f'D-tasks/models/d4_m={margin}_{s}.pth'

    print(f'Loading weights from {weights_path}')
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=map_location))

    return model
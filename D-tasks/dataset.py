# TODO: Implement the dataset class extending torch.utils.data.Dataset
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import random

test_path = 'D-tasks/data/test.pkl'
train_path = 'D-tasks/data/train.pkl'
meta_path = 'D-tasks/data/meta.pkl'

def open_files(train_path, test_path, meta_path):
    with open(train_path, 'rb') as f:
        train = pickle.load(f)

    with open(test_path, 'rb') as f:
        test = pickle.load(f)

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    return train, test, meta

def extracting_info(train, test, meta):
    data_train = train[b'data']
    fine_train = train[b'fine_labels']
    coarse_train = train[b'coarse_labels']

    data_test = test[b'data']
    fine_test = test[b'fine_labels']
    coarse_test = test[b'coarse_labels']

    fine_meta = meta['fine_label_names']
    coarse_meta = meta['coarse_label_names']

    return data_train, fine_train, coarse_train, data_test, fine_test, coarse_test, fine_meta, coarse_meta

def data_to_np(data):
    """Convert flat CIFAR arrays into (3,32,32) numpy arrays."""
    npdata = np.reshape(data, (-1, 3, 32, 32))
    return npdata

def to_tensor(data, fine_label, coarse_label):
    data = torch.tensor(data, dtype=torch.float32)
    fine_label = torch.tensor(fine_label, dtype=torch.long)
    coarse_label = torch.tensor(coarse_label, dtype=torch.long)
    return data, fine_label, coarse_label

class DatasetCreator(Dataset):
    """Dataset that supports transforms."""

    def __init__(self, features, fine_labels, coarse_labels, transform=None):
        assert len(features) == len(fine_labels), "Features and labels must have the same length."
        self.features = features
        self.fine_labels = fine_labels
        self.coarse_labels = coarse_labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        image = self.features[idx]  # (3,32,32) tensor (float32)
        fine_label = self.fine_labels[idx]
        coarse_label = self.coarse_labels[idx]

        # (C,H,W -> H,W,C)
        img_np = image.numpy().transpose(1, 2, 0).astype(np.uint8)

        if self.transform is not None:
            img_tensor = self.transform(img_np)
        else:
            img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1) / 255.

        return img_tensor, fine_label, coarse_label


def create_dataset(train_path, test_path, meta_path):
    train, test, meta = open_files(train_path, test_path, meta_path)

    (
        data_train, fine_train, coarse_train,
        data_test, fine_test, coarse_test,
        fine_meta, coarse_meta
    ) = extracting_info(train, test, meta)

    data_train = data_to_np(data_train)
    data_test = data_to_np(data_test)

    data_train, fine_train, coarse_train = to_tensor(
        data_train, fine_train, coarse_train
    )
    data_test, fine_test, coarse_test = to_tensor(
        data_test, fine_test, coarse_test
    )

    train_transform = TrainTransform()
    test_transform = TestTransform()

    dataset_train = DatasetCreator(
        data_train, fine_train, coarse_train, transform=train_transform
    )
    dataset_test = DatasetCreator(
        data_test, fine_test, coarse_test, transform=test_transform
    )

    return dataset_train, dataset_test

class TestTransform:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        img = img.float()

        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        if img.max() > 1.0:
            img = img / 255.0

        return img
    
class TrainTransform:
    def __call__(self, img):
        # Convert numpy to torch
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        img = img.float()

        # If image is HWC convert to CHW
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        if img.max() > 1.0:
            img = img / 255.0

        # Augmentations
        img = random_crop(img, size=32, padding=4)

        if random.random() < 0.5:
            img = hflip(img)

        if random.random() < 0.2:
            img = vflip(img)

        if random.random() < 0.5:
            img = random_rotate(img, max_angle=15)

        if random.random() < 0.7:
            img = color_jitter(
                img,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1
            )

        return img
    
def color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1):
    # Brightness
    if brightness > 0:
        factor = 1 + random.uniform(-brightness, brightness)
        img = img * factor

    # Contrast
    if contrast > 0:
        mean = img.mean(dim=(1, 2), keepdim=True)
        factor = 1 + random.uniform(-contrast, contrast)
        img = (img - mean) * factor + mean

    # Saturation
    if saturation > 0:
        gray = img.mean(dim=0, keepdim=True)
        factor = 1 + random.uniform(-saturation, saturation)
        img = (img - gray) * factor + gray

    return img.clamp(0.0, 1.0)

def random_rotate(img, max_angle=15):
    angle = random.uniform(-max_angle, max_angle)
    angle = np.deg2rad(angle)

    c, h, w = img.shape
    cy, cx = h // 2, w // 2

    yy, xx = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )

    yy = yy - cy
    xx = xx - cx

    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    y_rot = cos_a * yy - sin_a * xx + cy
    
    x_rot = sin_a * yy + cos_a * xx + cx

    y_rot = y_rot.round().long()
    x_rot = x_rot.round().long()

    mask = (
        (y_rot >= 0) & (y_rot < h) &
        (x_rot >= 0) & (x_rot < w)
    )

    out = torch.zeros_like(img)
    for ch in range(c):
        out[ch][mask] = img[ch][y_rot[mask], x_rot[mask]]

    return out

def hflip(img):
    return torch.flip(img, dims=[2])

def vflip(img):
    return torch.flip(img, dims=[1])

def random_crop(img, size=32, padding=4):
    # CHW format
    if padding > 0:
        img = torch.nn.functional.pad(
            img.unsqueeze(0),
            (padding, padding, padding, padding),
            mode="reflect"
        ).squeeze(0)

    _, h, w = img.shape
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)

    return img[:, top:top + size, left:left + size]


class ZeroshotDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict[b'data']                 
        self.fine_labels = data_dict[b'fine_labels']    
        self.coarse_labels = data_dict[b'coarse_labels']
        self.label_map = data_dict.get(b'label_map', None)
        self.transform = transform if transform is not None else TestTransform()

    def __len__(self):
        return len(self.fine_labels)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)
        fine_label = int(self.fine_labels[idx])
        coarse_label = int(self.coarse_labels[idx])

        img = self.transform(img)
        return img, fine_label, coarse_label

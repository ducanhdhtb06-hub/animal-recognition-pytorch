import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import ToTensor, Resize , Compose
from PIL import Image


class AnimaDataset(Dataset) :
    def __init__(self, root, train = True, transform = None) :
        self.transform = transform
        if train :
            mode = "train"
            self.root = os.path.join(root, mode)
        else :
            mode = "test"
            self.root = os.path.join(root, mode)
        self.categories = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
        self.images_path = []
        self.labels = []
        for i , category in enumerate( self.categories):
            data_file_path = os.path.join(self.root, category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name)
                self.images_path.append(file_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx) :
        image_path = self.images_path[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image , label

if __name__ =='__main__':
    root = "animals_v2/animals"
    transform = Compose([
        Resize((200, 200)),
        ToTensor(),

    ])
    training_data =  AnimaDataset(root="animals_v2/animals", train=True, transform=transform)
    training_loader = DataLoader(
            dataset=training_data,
            batch_size=16,
            num_workers=2,
            shuffle=True)
    for images, labels in training_loader:
            print(images.shape)
            print(labels)


import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import timm

import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
from tqdm import tqdm
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2 as cv


data_base_dir = Path.cwd().joinpath("data","sportData")
df = pd.read_csv(data_base_dir.joinpath("sports.csv"))
df.head()


df_train = df[df["dataset"]=="train"]
df_test = df[df["dataset"]=="test"]
df_valid = df[df["dataset"]=="valid"]
df_train.head()


train_img_dirs = [str(data_base_dir.joinpath(i)) for i in df_train["filepaths"].to_list()]
test_img_dirs = [str(data_base_dir.joinpath(i)) for i in df_test["filepaths"].to_list()]
valid_img_dirs = [str(data_base_dir.joinpath(i)) for i in df_valid["filepaths"].to_list()]


train_labels = df_train.labels.to_list()
test_labels = df_test.labels.to_list()
valid_labels = df_valid.labels.to_list()



unique_labels = df_train["labels"].unique()
DEVICE="mps" if torch.backends.mps.is_built() else "cpu"



class SPORTS(Dataset):
    def __init__(self, paths, labels, unique_labels):
        #get all necessary inputs like train directories and labels
        self.paths = paths
        self.labels = np.asarray(labels)
        self.unique_labels = np.asarray(unique_labels)


        
    def __len__(self,):
        #return len of the dataset
        return len(self.paths)
    
    def get_one_hot_encoding(self, cat):
        one_hot = np.asarray(cat == self.unique_labels)
        return one_hot
        
    def __getitem__(self, idx):
        #get data for one id value..pytorch will handle the batching for you!
    
        img_dir = self.paths[idx]
        label = self.labels[idx]

        img = cv.imread(img_dir)
        # img = cv.resize(img,(224,224))
        one_hot = self.get_one_hot_encoding(label)

        img = img.transpose((2,0,1)) #channel must come first 
        img = torch.tensor(img, dtype = torch.float)
        one_hot = torch.tensor(one_hot, dtype = torch.float)
        return img/255.0, one_hot
        


train_dataset = SPORTS(train_img_dirs,train_labels,unique_labels)
test_dataset = SPORTS(test_img_dirs,test_labels,unique_labels)
valid_dataset = SPORTS(valid_img_dirs,valid_labels,unique_labels)


train_dataloader = DataLoader(
    train_dataset,
    num_workers =2,
    batch_size = 32,
    shuffle = True
)
valid_dataloader = DataLoader(
    valid_dataset,
    num_workers =0,
    batch_size = 32,
    shuffle = True
)


class SportClassifier(nn.Module):
    def __init__(self, num_classes):
        #define necessary layers
        super().__init__()
        self.num_classes = num_classes      
        self.model = timm.create_model(model_name = "resnet34", pretrained = True)
        self.model.fc = nn.Linear(self.model.fc.in_features, out_features = num_classes)

        
    def forward(self,X):
        #define forward pass here
        return F.softmax(self.model(X), dim=-1)
    
model = SportClassifier(len(unique_labels)).to(DEVICE)

optimizer = Adam(lr = 0.001, params = model.parameters())

def loss_fn(y_pred, y_true):
    #define your loss function here e.g crossEntropyLoss
    y_pred = torch.clip(y_pred, 1e-8, 1-1e-8)
    l = y_true*torch.log(y_pred)
    l = l.sum(dim = -1)
    l = l.mean()
    return -l



def train_on_one_epoch(dataloader, optimizer, loss_fn, len_dataloader):
    #training for one epoch
    
    dataloader = tqdm(dataloader)
    L = 0
    acc = 0
    for i,(x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_pred = model(x)
        l = loss_fn(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        L+=l.item()
        acc+=np.sum(y_pred.cpu().detach().numpy().argmax(-1) == y.cpu().detach().numpy().argmax(-1))
    return L/len_dataloader, acc/len_dataloader


def valid_on_one_epoch(dataloader, loss_fn, len_dataloader):
    #validation for one epoch
    dataloader = tqdm(dataloader)
    L = 0
    acc = 0
    for i,(x, y) in enumerate(dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_pred = model(x)
        l = loss_fn(y_pred, y)
        L+=l.item()
        acc+=np.sum(y_pred.cpu().detach().numpy().argmax(-1) == y.cpu().detach().numpy().argmax(-1))
        
    return L/len_dataloader, acc/len_dataloader

def loadtimeChecker(dataloader):
    total = 0
    import time
    start = time.time()
    for i,(x, y) in enumerate(dataloader):
        end = time.time()
        duration = end-start
        total+=duration
        # print(f"Iteration: {i}\tTime taken: {(end-start)*10**3:.03f}ms")
        start = time.time()
    print(total)
    return 

if __name__ == '__main__':
    loadtimeChecker(train_dataloader)
    # prev_valid_acc = 0
    # for epoch in range(10):
    #     train_loss, train_acc = train_on_one_epoch(train_dataloader, optimizer, loss_fn, len(train_dataset))
    #     valid_loss, valid_acc = valid_on_one_epoch(valid_dataloader, loss_fn, len(valid_dataset))
    #     print(f"epoch : {epoch} | train loss : {train_loss} | valid_loss : {valid_loss} | train_acc : {train_acc} | valid_acc : {valid_acc}")
    #     if prev_valid_acc<valid_acc:
    #         print("model saved..!!")
    #         torch.save(model.state_dict(), "best.pt")
    #         prev_valid_acc = valid_acc

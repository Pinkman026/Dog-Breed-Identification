#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from models import get_resnet50
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.label import LabelEncoder
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from torch.autograd import Variable
import numpy as np
from torchvision import transforms
import os
from utils import DogDataset, get_train_dataset, train_epoch, val_epoch

BATCH_SIZE = 32
EPOCHS = 1
CUDA = torch.cuda.is_available()

data_dir = "E:/data/dog-breed-identification/"
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
data_train_csv = os.path.join(data_dir, 'labels.csv')
data_train_csv = pd.read_csv(data_train_csv)
filenames = data_train_csv.id.values
le = LabelEncoder()
labels = le.fit_transform(data_train_csv.breed)

filenames_train, filenames_val, labels_train, labels_val = \
    train_test_split(filenames, labels, test_size=0.1, stratify=labels)

dog_train = get_train_dataset(filenames_train, labels_train, BATCH_SIZE,
                              rootdir=train_dir)
dog_val = get_train_dataset(filenames_val, labels_val, BATCH_SIZE,
                            rootdir=train_dir)

net = get_resnet50(n_class=len(le.classes_))
criterion_train = nn.CrossEntropyLoss()
criterion_val = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.fc.parameters(), lr=0.0001)  # use default learning rate
state = {'val_acc': [], 'lives': 4, 'best_val_acc': 0}

if CUDA:
    net.cuda()
for epoch in range(EPOCHS):
    print("Epoch: ", epoch + 1)
    train_acc = train_epoch(net, dog_train, criterion_train, optimizer, CUDA)
    print("Evaluating...")
    val_acc = val_epoch(net, dog_val, criterion_val, CUDA)

    state['val_acc'].append(val_acc)
    if val_acc > state['best_val_acc']:
        state['lives'] = 4
        state['best_val_acc'] = val_acc
    else:
        state['lives'] -= 1
        print("Trial left :", state['lives'])
        if state['lives'] == 2:
            optimizer.param_groups[0]['lr'] /= 2
        if state['lives'] == 0:
            break

sample_result = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
print(sample_result.shape)

# test dataset definition
val_transform = transforms.Compose([transforms.Resize([224, 224]),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])
test_dataset = DogDataset(sample_result.id, labels=[0 for i in range(len(sample_result.id))], root_dir=test_dir,
                          transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(len(test_dataset))

net.eval()

# define sorfmax function to map output to probability
softmax = torch.nn.Softmax()

first = True
total_batch = int(len(test_dataset) / BATCH_SIZE)
for batch_idx, (x, y) in enumerate(test_loader):
    print("{} / {}".format(batch_idx, total_batch))
    if CUDA:
        x, y = x.cuda(), y.cuda()
    x = Variable(x)
    y = Variable(y)
    prediction = softmax(net(x)).detach().cpu().numpy()
    if first:
        results = prediction
        first = False
    else:
        results = np.concatenate((results, prediction), axis=0)

df = pd.DataFrame(data=results, index=sample_result.id, columns=le.classes_)
df.to_csv("submit1.csv")

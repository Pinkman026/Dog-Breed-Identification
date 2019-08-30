#!/usr/bin/python
# -*- coding: utf-8 -*-


from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def imshow(axis, inp):
    """Denormalize and show"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    axis.imshow(inp)

class DogDataset(Dataset):
    """Dog Breed Dataset"""

    def __init__(self, filenames, labels, root_dir, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        label = self.labels[item]
        img_name = os.path.join(self.root_dir, self.filenames[item] + '.jpg')

        with Image.open(img_name) as f:
            img = f.convert('RGB')

        if self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img, self.filenames[item]
        else:
            return img, self.labels[item]


def get_train_dataset(filenames, labels, batch_size, rootdir='data/train'):
    composed = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                   ])
    dog_trainset = DogDataset(filenames, labels, transform=composed, root_dir=rootdir)
    dog_train = DataLoader(dog_trainset, batch_size, True)
    return dog_train


def get_test_dataset(filenames, batch_size, rootdir='data/test'):
    composed = transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                                   ])
    dog_testset = DogDataset(filenames, None, transform=composed, root_dir=rootdir)
    dog_test = DataLoader(dog_testset, batch_size, False)
    return dog_test


def train_epoch(net, data_iter, criterion, optimizer, use_cuda, print_every=20):
    net.eval()  # net.train()
    correct = 0
    for batch_idx, (x, y) in enumerate(data_iter):
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        x = Variable(x)
        y = Variable(y)
        optimizer.zero_grad()
        logits = net(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        prediction = torch.argmax(logits, 1)
        cur_correct = (prediction == y).sum().float()
        cur_accuracy = cur_correct / x.shape[0]
        correct += cur_correct

        if batch_idx % print_every == 0:
            print('current batch: {}/{} ({:.0f}%)\tLoss: {:.6f}\tAcc: {:.6f}'.format(
                batch_idx, len(data_iter),
                100. * batch_idx / len(data_iter), loss.data.item(), cur_accuracy))

    accuracy = correct / len(data_iter.dataset)
    print('Train epoch Acc: {:.6f}'.format(accuracy))
    return accuracy


def val_epoch(net, data_iter, criterion, use_cuda):
    test_loss = 0
    correct = 0
    net.eval()
    for batch_idx, (x, y) in enumerate(data_iter):
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        x = Variable(x)
        y = Variable(y)
        logits = net(x)
        loss = criterion(logits, y)

        test_loss += loss.data.item()
        prediction = torch.argmax(logits, 1)
        cur_correct = (prediction == y).sum().float()
        correct += cur_correct

    test_loss /= len(data_iter.dataset)
    accuracy = correct / len(data_iter.dataset)

    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(data_iter.dataset), 100. * accuracy))

    return accuracy

def visualize_model(dataloders, model, num_images=16, use_gpu=True):
    cnt = 0
    fig = plt.figure(1, figsize=(16, 16))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.05)
    for i, (inputs, labels) in enumerate(dataloders['valid']):
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            ax = grid[cnt]
            imshow(ax, inputs.cpu().data[j])
            ax.text(10, 210, '{}/{}'.format(preds[j], labels.data[j]),
                    color='k', backgroundcolor='w', alpha=0.8)
            cnt += 1
            if cnt == num_images:
                return



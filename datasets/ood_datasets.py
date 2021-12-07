import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
import random
import math
import numbers
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

DATA_PATH = '../GridRot/datasets/data/'
IMAGENET_PATH = '../GridRot/datasets/data/ImageNet'
# DATA_PATH = '/mnt/hdd0/WJ/OOD_labeled/'
# IMAGENET_PATH = '/mnt/hdd0/WJ/OOD_labeled/'



CIFAR10_SUPERCLASS = list(range(10))  # one class
IMAGENET_SUPERCLASS = list(range(30))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]


class Shifting(nn.Module):
    def __init__(self, direction=4, max_pixel=7, min_pixel=2, fix_pixel=4):
        super(Shifting, self).__init__()
        self.direction = direction
        self.max_pixel = max_pixel
        self.min_pixel = min_pixel
        self.fix_pixel = fix_pixel

    def forward(self, input, aug_index=None):
        # _device = input.device

        # _, _, H, W = input.size()
        aug_index = np.random.randint(self.direction)

        if self.fix_pixel is not None:
            pixel_index = self.fix_pixel
        else:
            pixel_index = np.random.randint(self.min_pixel, self.max_pixel)
        # output = torch.zeros_like(input)
        input2 = np.zeros_like(input)
        input = np.asarray(input)#.numpy()
        # print(input)
        # print(input.shape)
        if aug_index == 0:
            # temp = input[:, :pixel_index, :]
            # output[:, :-pixel_index, :] = input[:, pixel_index:, :]
            # output[:, -pixel_index:, :] = 0#temp
            #
            # temp = output[:, :, :pixel_index]
            # input2[:, :, :-pixel_index] = output[:, :, pixel_index:]
            # input2[:, :, -pixel_index:] = 0#temp
            input2 = cv2.resize(input[pixel_index:,pixel_index:,:],dsize=(32,32))
        elif aug_index == 1:
            # temp = input[:, :, :pixel_index]
            # output[:, :, :-pixel_index] = input[:, :, pixel_index:]
            # output[:, :, -pixel_index:] = 0#temp
            #
            # ### Up
            # temp = output[:, :, :pixel_index]
            # input2[:, :, :-pixel_index] = output[:, :, pixel_index:]
            # input2[:, :, -pixel_index:] = 0#temp
            input2 = cv2.resize(input[pixel_index:, :-pixel_index, :], dsize=(32, 32))
        elif aug_index == 2:
            # temp = input[:, -pixel_index:, :]
            # output[:, pixel_index:, :] = input[:, :-pixel_index, :]
            # output[:, :pixel_index, :] = 0#temp
            #
            # ### Down
            # temp = output[:, :, -pixel_index:]
            # input2[:, :, pixel_index:] = output[:, :, :-pixel_index]
            # input2[:, :, :pixel_index] = 0#temp
            input2 = cv2.resize(input[:-pixel_index, pixel_index:, :], dsize=(32, 32))
        elif aug_index == 3:
            # temp = input[:, :, -pixel_index:]
            # output[:, :, pixel_index:] = input[:, :, :-pixel_index]
            # output[:, :, :pixel_index] = 0#temp
            #
            # ### Down
            # temp = output[:, :, -pixel_index:]
            # input2[:, :, pixel_index:] = output[:, :, :-pixel_index]
            # input2[:, :, :pixel_index] = 0#temp
            input2 = cv2.resize(input[:-pixel_index, :-pixel_index, :], dsize=(32, 32))
        # input2 = torch.Tensor(input2)
        return input2

class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)




def get_transform(image_size=None, shifting=False):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here
    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose([
            Shifting(),
            transforms.ToPILImage(),
            transforms.Resize((image_size[0], image_size[1])),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),

        ])
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        shift_test_transform = transforms.Compose([
            Shifting(),
            transforms.ToPILImage(),
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),

        ])

    else:  # use default image size
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_transform = transforms.ToTensor()
    if shifting:
        return train_transform, test_transform, shift_test_transform
    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


def get_dataset(P, dataset, test_only=False, image_size=None, download=True, eval=False, shifting=False):
    if dataset in ['imagenet', 'cub', 'stanford_dogs', 'flowers102',
                   'places365', 'food_101', 'caltech_256', 'dtd', 'pets']:
        train_transform, test_transform = get_transform_imagenet()
    else:
        # if shifting:
        train_transform, test_transform, shift_test_transform = get_transform(image_size=image_size, shifting=True)
        # train_transform, test_transform = get_transform(image_size=image_size)

    if dataset == 'cifar10':
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=test_transform)
        shift_test_set = datasets.CIFAR10(DATA_PATH, train=False, download=download, transform=shift_test_transform)

    elif dataset == 'cifar100':
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(DATA_PATH, train=True, download=download, transform=train_transform)
        test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=test_transform)
        shift_test_set = datasets.CIFAR100(DATA_PATH, train=False, download=download, transform=shift_test_transform)
    elif dataset == 'svhn':
        assert test_only and image_size is not None
        test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=test_transform)
        shift_test_set = datasets.SVHN(DATA_PATH, split='test', download=download, transform=shift_test_transform)

    elif dataset == 'lsun_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        shift_test_set = datasets.ImageFolder(test_dir, transform=shift_test_transform)

    elif dataset == 'lsun_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'LSUN_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        shift_test_set = datasets.ImageFolder(test_dir, transform=shift_test_transform)

    elif dataset == 'imagenet_resize':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_resize')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        shift_test_set = datasets.ImageFolder(test_dir, transform=shift_test_transform)

    elif dataset == 'imagenet_fix':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'Imagenet_fix')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        shift_test_set = datasets.ImageFolder(test_dir, transform=shift_test_transform)

    elif dataset == 'imagenet':
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, 'one_class_train')
        test_dir = os.path.join(IMAGENET_PATH, 'one_class_test')
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == 'stanford_dogs':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'stanford_dogs')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'cub':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'cub200')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'flowers102':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'flowers102')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'places365':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'places365')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'food_101':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'food-101/food-101/food-101', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'caltech_256':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'caltech-256')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'dtd':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'dtd/dtd', 'images')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == 'pets':
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, 'pets')
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()



    if test_only:
        if shifting:
            return shift_test_set
        return test_set


    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == 'cifar10':
        return CIFAR10_SUPERCLASS
    elif dataset == 'cifar100':
        return CIFAR100_SUPERCLASS
    elif dataset == 'imagenet':
        return IMAGENET_SUPERCLASS
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):

    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=resize_scale),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    clean_trasform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform



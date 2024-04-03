# import cv2
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms, datasets

NUM_DATASET_WORKERS = 4
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def get_loader(config):
    if config.dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root=config.train_data_dir,
                                         train=True,
                                         transform=transform_train,
                                         download=False)

        test_dataset = datasets.CIFAR10(root=config.test_data_dir,
                                        train=False,
                                        transform=transform_test,
                                        download=False)
    elif config.dataset == "DIV2K":
        transform_train = transforms.Compose([
            transforms.RandomCrop((config.image_dims[1], config.image_dims[2])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.CenterCrop((config.image_dims[1], config.image_dims[2])),
            transforms.ToTensor()])

        train_dataset = datasets.ImageFolder(root=config.train_data_dir,
                                             transform=transform_train, )

        test_dataset = datasets.ImageFolder(root=config.test_data_dir,
                                            transform=transform_test)
    elif config.dataset == "CelebA":
        transform_train = transforms.Compose([
            transforms.RandomCrop((config.image_dims[1], config.image_dims[2])),
            transforms.ToTensor()])

        transform_test = transforms.Compose([
            transforms.CenterCrop((config.image_dims[1], config.image_dims[2])),
            transforms.ToTensor()])
        train_dataset = datasets.ImageFolder(root=config.train_data_dir,
                                             transform=transform_train)

        test_dataset = datasets.ImageFolder(root=config.test_data_dir,
                                            transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               num_workers=NUM_DATASET_WORKERS,
                                               pin_memory=True,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               shuffle=True,
                                               drop_last=True)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=config.test_batch,
                                  shuffle=False)

    return train_loader, test_loader


class config():
    dataset = "CelebA"
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    device_ids = [0]
    if_sample = False
    # logger
    print_step = 39
    plot_step = 10000
    # filename = datetime.now().__str__()[:-16]
    models = 'E:\code\DDPM\SemDiffusion\Autoencoder\history'
    logger = None
    equ = "MMSE"
    # training details
    normalize = False
    learning_rate = 0.0001
    epoch = 20

    save_model_freq = 20
    if dataset == "CIFAR10":
        image_dims = (3, 32, 32)
        train_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
        test_data_dir = r"E:\code\DDPM\DenoisingDiffusionProbabilityModel-ddpm--main\DenoisingDiffusionProbabilityModel-ddpm--main\CIFAR10"
    elif dataset == "DIV2K":
        image_dims = (3, 256, 256)
        train_data_dir = r"D:\dateset\DIV2K\DIV2K_train_HR"
        test_data_dir = r"D:\dateset\DIV2K\DIV2K_valid_HR"
    elif dataset == "CelebA":
        image_dims = (3, 128, 128)
        train_data_dir = r"D:\dateset\CelebA\Img\trainset"
        test_data_dir = r"D:\dateset\CelebA\Img\validset"
    batch_size = 1
    # batch_size = 100
    downsample = 4


if __name__ == "__main__":
    train_loader, test_loader = get_loader(config)
    image = next(iter(train_loader))[0]
    print(image)

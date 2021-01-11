import os
import torch
import numpy as np
import pandas as pd
from skimage import io
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class TerrainDataset(Dataset):
    """TerrainDataset"""

    def __init__(self, img_dir, transform=None):

        self.img_dir = img_dir
        self.transform = transform
        self.data_list = os.listdir(img_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.data_list[idx])
        image = io.imread(img_name)
        # image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR cv2.IMREAD_GRAYSCALE
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = io.imread(img_name) # (218, 178, 3) (H, W, C) - numpy.ndarray - dtype=uint8 [0, 255]

        sample = image
        if self.transform:
            sample = self.transform(sample)
        return sample

def get_dataloader(batch_size, num_workers):
    trn_dataset = TerrainDataset(
                    img_dir='data/terrain/train_images/',
                    transform=transforms.Compose([  
                                transforms.ToPILImage(),
                                transforms.Resize((64, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                    ]),
                )
    eval_dataset = TerrainDataset(
                    img_dir='data/terrain/test_images/',
                    transform=transforms.Compose([  
                                transforms.ToPILImage(),
                                transforms.Resize((64, 64)),
                                transforms.ToTensor(),
                    ]),
                )
    trn_loader = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
    return trn_dataset, eval_dataset, trn_loader, eval_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = TerrainDataset(
                    img_dir='data/terrain/train_images/',
                    transform=transforms.Compose([  
                                transforms.ToPILImage(),
                                # transforms.CenterCrop(128),
                                transforms.Resize((64, 64)),
                                transforms.RandomHorizontalFlip(p=0.5),
                                # transforms.RandomVerticalFlip(p=0.5),
                                # transforms.ToTensor(),
                    ]),
                    )


    # for i in range(10):
        # print(dataset[i])
    print(dataset[33])
    # # print(dataset[0].shape)
    plt.imshow(dataset[33])
    plt.show()
    # import ipdb
    # ipdb.set_trace()
    # dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4, drop_last=True)
    # for i, mini_batch in enumerate(dataloader):
        # print(f"i: {i}, mini_batch.size(): {mini_batch.size()}")

    # import ipdb
    # ipdb.set_trace()


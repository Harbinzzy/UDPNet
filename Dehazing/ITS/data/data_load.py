import os
import torch
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader

def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader

def valid_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test')),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'gt', self.image_list[idx].split('_')[0]+'.png'))

        file_name = os.path.splitext(self.image_list[idx])[0]
        depth_image_path = os.path.join(self.image_dir, 'depth2l', file_name + '.png')
        depth = Image.open(depth_image_path).convert("L")
        # depth = Image.new("L", depth.size, 128)

        if self.transform:
            image, depth, label = self.transform(image, depth, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            depth = F.to_tensor(depth)

        image = torch.cat([image, depth], dim=0)

        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError

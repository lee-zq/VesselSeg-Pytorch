from torch.utils.data import Dataset
import torch
import numpy as np

from .MyTransforms import *

class TestDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs):
        self.imgs = patches_imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx,...]).float()


class TrainDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs,patches_masks,mode="train"):

        self.imgs = patches_imgs
        self.masks = patches_masks
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                # RandomResize([56,72],[56,72]),
                # RandomCrop((48, 48)),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
                RandomRotate()
            ])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        mask = self.masks[idx]
        data = self.imgs[idx]

        data = torch.from_numpy(data).float()
        mask = torch.from_numpy(mask).long()

        if self.transforms:
            data, mask = self.transforms(data, mask)
        return data, mask.squeeze(0)

"""
class TrainDataset(Dataset):
    def __init__(self, patches_imgs,patches_masks_train):
        self.imgs = patches_imgs
        self.masks = patches_masks_train

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        tmp = self.masks[idx]
        tmp = np.squeeze(tmp,0)
        return torch.from_numpy(self.imgs[idx,...]).float(), torch.from_numpy(tmp).long()
"""

class TrainDataset_imgaug(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs,patches_masks_train):
        self.imgs = patches_imgs
        self.masks = patches_masks_train
        self.seq = iaa.Sequential([
            # iaa.Sharpen((0.1, 0.5)),
            iaa.flip.Fliplr(p=0.5),
            iaa.flip.Flipud(p=0.5),
            # sharpen the image
            # iaa.GaussianBlur(sigma=(0.0, 0.1)),  # apply water effect (affects heatmaps)
            # iaa.Affine(rotate=(-20, 20)),
            # iaa.ElasticTransformation(alpha=16, sigma=8),   # water-like effect
            ], random_order=True)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        mask = self.masks[idx,0]
        data = self.imgs[idx]
        data = data.transpose((1,2,0))
        mask = ia.SegmentationMapsOnImage(mask, shape=data.shape)

        # 这里可以通过加入循环的方式，对多张图进行数据增强。
        seq_det = self.seq.to_deterministic()  # 确定一个数据增强的序列
        data = seq_det.augment_image(data).transpose((2,0,1))/255.0  # 将方法应用在原图像上
        mask = seq_det.augment_segmentation_maps([mask])[0].get_arr().astype(np.uint8)

        return torch.from_numpy(data).float(), torch.from_numpy(mask).long()

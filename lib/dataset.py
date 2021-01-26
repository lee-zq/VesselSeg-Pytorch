"""
This part is based on the dataset class implemented by pytorch, 
including train_dataset and test_dataset, as well as data augmentation
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

class TrainDataset(Dataset):
    def __init__(self, patches_imgs,patches_masks,mode="train"):
        self.imgs = patches_imgs
        self.masks = patches_masks
        self.transforms = None
        if mode == "train":
            self.transforms = Compose([
                # RandomResize([56,72],[56,72]),
                RandomCrop((48, 48)),
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

#----------------------data augment-------------------------------------------
class Resize:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape

    def __call__(self, img, mask):
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].byte()

class RandomResize:
    def __init__(self, w_rank,h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank

    def __call__(self, img, mask):
        random_w = random.randint(self.w_rank[0],self.w_rank[1])
        random_h = random.randint(self.h_rank[0],self.h_rank[1])
        self.shape = [random_w,random_h]
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], mask[0].long()

class RandomCrop:
    def __init__(self, shape):
        self.shape = [shape, shape] if isinstance(shape, int) else shape
        self.fill = 0
        self.padding_mode = 'constant'

    def _get_range(self, shape, crop_shape):
        if shape == crop_shape:
            start = 0
        else:
            start = random.randint(0, shape - crop_shape)
        end = start + crop_shape
        return start, end

    def __call__(self, img, mask):
        _, h, w = img.shape
        sh, eh = self._get_range(h, self.shape[0])
        sw, ew = self._get_range(w, self.shape[1])
        return img[:, sh:eh, sw:ew], mask[:, sh:eh, sw:ew]

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(1)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img,cnt,[1,2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0,self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)


class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return normalize(img, self.mean, self.std, False), mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class TestDataset(Dataset):
    """Endovis 2018 dataset."""

    def __init__(self, patches_imgs):
        self.imgs = patches_imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.imgs[idx,...]).float()

#----------------------image aug--------------------------------------
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
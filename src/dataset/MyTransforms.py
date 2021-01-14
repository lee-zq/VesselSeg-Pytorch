import random
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

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


# class RandomFlip:
#     def __init__(self, prob=0.5, side=(True, True)):
#         self.prob = prob
#         self.side = side

#     def _flip(self, img, prob):
#         if self.side[0] and prob[0] <= self.prob:
#             img = img.flip(1)
#         if self.side[1] and prob[1] <= self.prob:
#             img = img.flip(2)
#         return img

#     def __call__(self, img, mask):
#         prob = (random.uniform(0, 1), random.uniform(0, 1))
#         return self._flip(img, prob), self._flip(mask, prob)

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
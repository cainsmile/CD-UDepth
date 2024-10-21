import os 
import pandas as pd
import random
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class TestDataset(Dataset):
    def __init__(self, root_dir, tranest_dir, transform=None):
        self.root_dir = root_dir
        self.tranest_dir = tranest_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def RGB_to_IMT(self, image, trans):
        r, g, b = image.split()
        r = np.array(r)
        gb_max = np.maximum.reduce([np.array(g), np.array(b)])
        gray_c = np.array(ImageOps.grayscale(image))
        combined = np.stack((np.array(trans),gray_c,gb_max), axis=-1)
        return Image.fromarray(combined)

    def __getitem__(self, idx):
        img_fn = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])
        tranest_fn = os.path.join(self.tranest_dir, os.listdir(self.root_dir)[idx])

        image = Image.open(img_fn)
        
        trans = Image.open(tranest_fn)
        trans = self.RGB_to_IMT(image,trans)

        img_fn = img_fn.split("/")[-1]

        sample = {'image': image, 'trans':trans,'file_name': img_fn}

        img_fn = img_fn.split("/")[-1]

        if self.transform:  
            sample = {'image': self.transform(image), 'trans': self.transform(trans),'file_name': img_fn}
        return sample

class ToTestTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, image):
        image = image.resize((640, 480))
        image = self.to_tensor(image)
        return image

    def to_tensor(self, pic):
        pic = np.array(pic)
        if not (_is_numpy_image(pic) or _is_pil_image(pic)):
            raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
                             
        if isinstance(pic, np.ndarray):
            if pic.ndim == 2:
                pic = pic[..., np.newaxis]
                
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(255)


# This file is mostly taken from BTS; author: Jin Han Lee, with only slight modifications

import os
import random

import numpy as np
import torch
import torch.utils.data.distributed
from torch.nn import MaxPool2d
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class DepthDataLoader(object):
    def __init__(self, config, mode):
        self._early_stopping_patience = config._early_stopping_patience
        if mode == 'train':
            self.data_samples = DataLoadPreprocess(config, mode, transform=preprocessing_transforms(mode))
            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.data_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.data_samples, config.train_batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.data_samples = DataLoadPreprocess(config, mode, transform=preprocessing_transforms(mode))
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.data_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.data_samples = DataLoadPreprocess(config, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.data_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
    
    @property
    def early_stopping_patience(self):
        return self._early_stopping_patience[self.data_samples.current_strategy]

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None):
        self.config = config
        if mode == 'online_eval':
            with open(config.test_filenames_file, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.train_filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.strategies = config.strategies
        self.current_strategy = 0
        self.multiple_strategy = config.multiple_strategy

        self.data_path = config.input_data_path 
        self.gt_path = config.groundtruth_data_path
        
        self.do_kb_crop = config.do_kb_crop
        self.dataset = config.dataset
        self.do_random_rotate = config.do_random_rotate
        self.rotation_degree = config.rotation_degree
        self.input_height = config.input_height
        self.input_width = config.input_width

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        image_path = os.path.join(self.data_path, remove_leading_slash(sample_path.split()[0]))
        depth_path = os.path.join(self.gt_path, remove_leading_slash(sample_path.split()[1]))
        image = Image.open(image_path)
        depth_gt = Image.open(depth_path)
        height = image.height
        width = image.width
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)

        if self.do_kb_crop:
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth_gt = depth_gt[top_margin:top_margin + 352,left_margin:left_margin + 1216]
            image = image[top_margin:top_margin + 352,left_margin:left_margin + 1216,:]

        if self.mode == 'train':
            # To avoid blank boundaries due to pixel registration
            if self.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if self.do_random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.rotation_degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                
            image, depth_gt = self.random_crop(image, depth_gt, self.input_height, self.input_width)
            image, depth_gt = self.train_augment(image, depth_gt)
            if self.multiple_strategy:
                depth_gt = self.dilation(depth_gt,**self.strategies[self.current_strategy])

        if self.dataset == 'nyu':
            depth_gt = depth_gt / 1000.0
        else:
            depth_gt = depth_gt / 256.0

        sample = {'image': image, 'depth': depth_gt}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def dilation(self,depth_gt,pool_size=(2,2),iterations=1): # TODO
        if iterations> 0:
            for _ in range(iterations):
                depth_gt = MaxPool2d(kernel_size=pool_size)(depth_gt)
        print(depth_gt.shape)
        return depth_gt

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_augment(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
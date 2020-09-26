import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import folder
import torch.utils.data
import torch.utils.data.distributed

from .transforms import TwoCropsTransform, GaussianBlur

data_path_dict = {
    'imagenet': '/scr-ssd/datasets/imagenet',
    'tiny-imagenet': '/scr/colinwei/data/tiny-imagenet'
}
crop_size_dict = {
    'imagenet': 224,
    'tiny-imagenet': 64
}
resize_size_dict = {
    'imagenet': 256,
    'tiny-imagenet': 74
}
num_classes_dict = {
    'imagenet': 1000,
    'tiny-imagenet': 200
}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def obtain_aug(dataset, data_aug): 
    crop_size = crop_size_dict[dataset]   
    if data_aug == 'standard':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif data_aug == 'mocov1':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif data_aug == 'mocov2':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif data_aug == 'weak':
        # weaker augmentation that doesn't do as big of a crop
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_size, scale=(0.8, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else: # data_aug is None
        train_transform = transforms.Compose([
            transforms.Resize(resize_size_dict[dataset]),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])
    return train_transform

def load_train(
    dataset, num_per_class, distributed, 
    batch_size, workers, alt_aug=None, 
    data_aug='pretrain', mode='train'):
    '''
    data_aug:
        if pretrain, apply contrastive learning data augmentation (returning 2 crops),
        if standard, simply choose a single random crop (for linear classification).
        if off, choose center crop (no data augmentation applied).
    '''

    
    data_path = data_path_dict[dataset]
    assert mode in ['train', 'val']
    traindir = os.path.join(data_path, mode)
    
    train_transform = obtain_aug(dataset, data_aug)
    if alt_aug is not None:
        alt_aug = obtain_aug(dataset, alt_aug)
    train_dataset = SubsetImageFolder(
        traindir, alt_transform=alt_aug, 
        transform=train_transform, num_per_class=num_per_class)
    print('train dataset size is', len(train_dataset))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(not distributed),
        num_workers=workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    
    return train_sampler, train_loader

def load_val(dataset, batch_size, workers):
    valdir = os.path.join(data_path_dict[dataset], 'val')
    return torch.utils.data.DataLoader(
        folder.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(resize_size_dict[dataset]),
            transforms.CenterCrop(crop_size_dict[dataset]),
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

class SubsetImageFolder(folder.DatasetFolder):
    """
    Data loader that loads only a subset of the samples
    """
    def __init__(self, root, alt_transform=None, 
                 transform=None, target_transform=None, num_per_class=None,
                 loader=folder.default_loader, extensions=folder.IMG_EXTENSIONS):
        super(folder.DatasetFolder, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, num_per_class)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions
        self.alt_transform = alt_transform

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        if self.alt_transform is None:
            return super(SubsetImageFolder, self).__getitem__(index)
        else:
            path, target = self.samples[index]
            orig_sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(orig_sample)
            else:
                sample = orig_sample
            if self.target_transform is not None:
                target = self.target_transform(target)
            return (sample, self.alt_transform(orig_sample)), target

def make_dataset(directory, class_to_idx, extensions, num_per_class):
    instances = []
    directory = os.path.expanduser(directory)
    def is_valid_file(x):
        return folder.has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        num_added = 0
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            if num_added >= num_per_class:
                break
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    num_added += 1
                    if num_added >= num_per_class:
                        break
    return instances

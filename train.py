import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol_pytorch import BYOL
import pytorch_lightning as pl

# test model, a resnet 50

resnet = models.resnet50(num_classes=200)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--image_folder', type=str, default='/scr/colinwei/data/tiny-imagenet/train',
    help='path to your folder of images for self-supervised learning')
parser.add_argument('--batch_size', type=int, default=256, help='should try largest')
parser.add_argument('--epochs', type=int, default=1000, help='epochs')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--save')

args = parser.parse_args()

# constants

LR = args.lr
NUM_GPUS = 1
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
IMAGE_SIZE = 64 # from crop size dict
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)
        self.i = 0

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, batch_idx):
        loss = self.forward(images)
        if batch_idx == 0:
            torch.save(self.learner.state_dict(), 'ckpt' + args.save + '-' + str(self.i) + '.pt')
            self.i += 1
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        return self.transform(img)

# main

if __name__ == '__main__':
    ds = ImagesDataset(args.image_folder, IMAGE_SIZE)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 128,
        projection_hidden_size = 2048,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(gpus=NUM_GPUS, max_epochs=EPOCHS)
    trainer.fit(model, train_loader)

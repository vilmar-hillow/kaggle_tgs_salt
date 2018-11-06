import argparse
import json
from pathlib import Path
from validation import validation_binary

import torch
import cv2
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.backends.cudnn

from models.resnext import ResNext
from models.wideresnet import WideResnet
from models.wideresnet_short import WideResnetShort
from loss import LossBinary, LossLovasz, LossStableBCE
from dataset import SaltDataset
import utils

import sys

from prepare_train_val import get_split

from albumentations import (
    Flip,
    OneOf,
    RandomRotate90,
    ShiftScaleRotate,
    HorizontalFlip,
    Normalize,
    MotionBlur,
    MedianBlur,
    Blur,
    CLAHE,
    IAASharpen,
    IAAEmboss,
    PadIfNeeded,
    RandomContrast,
    RandomBrightness,
    ElasticTransform,
    HueSaturationValue,
    RGBShift,
    Compose,
    Resize
)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', default=0.5, type=float)
    arg('--device-ids', type=str, default='0', help='For example 0,1 to run on two GPUs')
    arg('--fold', type=int, help='fold', default=0)
    arg('--root', default='runs/debug', help='checkpoint root')
    arg('--batch-size', type=int, default=1)
    arg('--n-epochs', type=int, default=100)
    arg('--lr', type=float, default=0.001)
    arg('--workers', type=int, default=8)
    arg('--loss', type=str, default='BCE', choices=['BCE', 'StableBCE', 'Lovasz'])
    arg('--model', type=str, default='ResNext', choices=['WideResnet', 'WideResnetShort', 'ResNext'])

    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    if args.model == 'ResNext':
        model = ResNext(pretrained=True)
    elif args.model == 'WideResnet':
        model = WideResnet(pretrained=True)
    elif args.model == 'WideResnetShort':
        model = WideResnetShort(pretrained=True)

    print('CUDA: {}'.format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        if args.device_ids:
            device_ids = list(map(int, args.device_ids.split(',')))
        else:
            device_ids = None
        model = nn.DataParallel(model, device_ids=device_ids).cuda()

    if args.loss == "Lovasz":
        loss = LossLovasz()
    elif args.loss == 'StableBCE':
        loss = LossStableBCE(jaccard_weight=args.jaccard_weight)
    else:
        loss = LossBinary(jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    def make_loader(file_names, shuffle=False, transform=None, batch_size=1):
        return DataLoader(
            dataset=SaltDataset(file_names, transform=transform),
            shuffle=shuffle,
            num_workers=args.workers,
            batch_size=batch_size,
            pin_memory=torch.cuda.is_available()
        )

    train_file_names, val_file_names = get_split(args.fold)

    print('num train = {}, num_val = {}'.format(len(train_file_names), len(val_file_names)))

    def train_transform(p=1):
        return Compose([
            # ShiftScaleRotate(p=0.5),
            HorizontalFlip(p=0.5),
            Blur(blur_limit=3, p=.5),
            RandomContrast(p=0.3),
            RandomBrightness(p=0.3),
            ElasticTransform(p=0.3),
            Resize(202, 202, interpolation=cv2.INTER_NEAREST),
            PadIfNeeded(256, 256),
            Normalize(p=1)
        ], p=p)

    def val_transform(p=1):
        return Compose([
            Resize(202, 202, interpolation=cv2.INTER_NEAREST),
            PadIfNeeded(256, 256),
            Normalize(p=1)
        ], p=p)

    train_loader = make_loader(train_file_names, shuffle=True, transform=train_transform(p=1),
                               batch_size=args.batch_size)
    valid_loader = make_loader(val_file_names, transform=val_transform(p=1),
                               batch_size=args.batch_size)

    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    valid = validation_binary

    utils.train(
        init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
        args=args,
        model=model,
        criterion=loss,
        train_loader=train_loader,
        valid_loader=valid_loader,
        validation=valid,
        fold=args.fold
    )


if __name__ == '__main__':
    main()

"""
Script generates predictions, splitting original images into tiles, and assembling prediction back together
"""
import argparse
from prepare_train_val import get_split, get_test, get_split_old
from prepare_data import original_height, original_width
from dataset import SaltDataset
import cv2
from models.resnext import ResNext
from models.wideresnet import WideResnet
from models.wideresnet_short import WideResnetShort
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import utils
import prepare_data
from torch.utils.data import DataLoader
from torch.nn import functional as F
from albumentations import Compose, Normalize, PadIfNeeded, Resize
from albumentations.augmentations.functional import center_crop, resize


def img_transform(p=1):
    return Compose([
        Resize(202, 202, interpolation=cv2.INTER_NEAREST),
        PadIfNeeded(256, 256),
        Normalize(p=1)
    ], p=p)


def get_model(model_path, model_type='ResNext'):
    """

    :param model_path:
    :param model_type: 'ResNext', 'WideResnet', 'WideResnetShort'
    :return:
    """

    if model_type == 'ResNext':
        model = ResNext()
    elif model_type == 'WideResnet':
        model = WideResnet()
    elif model_type == 'WideResnetShort':
        model = WideResnetShort()

    state = torch.load(str(model_path))
    state = {key.replace('module.', ''): value for key, value in state['model'].items()}
    if model_type == 'ResNext':
        state = {key.replace('se_', 'se_module.'): value for key, value in state.items()}
    model.load_state_dict(state)

    if torch.cuda.is_available():
        return model.cuda()

    model.eval()

    return model


def predict(model, from_file_names, batch_size, to_path, img_transform):
    loader = DataLoader(
        dataset=SaltDataset(from_file_names, transform=img_transform, mode='predict'),
        shuffle=False,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    with torch.no_grad():
        for _, (inputs, paths) in enumerate(tqdm(loader, desc='Predict')):
            inputs = utils.cuda(inputs)

            outputs = model(inputs)

            for i, _ in enumerate(paths):
                factor = prepare_data.binary_factor
                t_mask = (F.sigmoid(outputs[i, 0]).data.cpu().numpy() * factor).astype(np.uint8)
                final_mask = center_crop(t_mask, 202, 202)
                final_mask = resize(final_mask, original_height, original_width, interpolation=cv2.INTER_AREA)

                to_path.mkdir(exist_ok=True, parents=True)

                cv2.imwrite(str(to_path / (Path(paths[i]).stem + '.png')), final_mask)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_path', type=str, default='runs/resnext', help='path to model folder')
    arg('--model_type', type=str, default='ResNext', help='network architecture',
        choices=['WideResnet', 'WideResnetShort', 'ResNext'])
    arg('--output_path', type=str, help='path to save images', default='predictions')
    arg('--batch-size', type=int, default=32)
    arg('--fold', type=int, default=-1, choices=[0, 1, 2, 3, 4, -1, -2], help='-1: all folds')
    arg('--final', action='store_true')
    arg('--workers', type=int, default=4)

    args = parser.parse_args()

    if args.final:
        file_names = get_test()
        for fold in [0, 1, 2, 3, 4]:
            model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=fold))),
                              model_type=args.model_type)
            print('num file_names = {}'.format(len(file_names)))
            output_path = Path(args.output_path) / Path(str(fold))
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, img_transform=img_transform(p=1))
    elif args.fold == -2:
        for fold in [0, 1, 2, 3]:
            _, file_names = get_split_old(fold)
            model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=fold))),
                              model_type=args.model_type)

            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path)
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, img_transform=img_transform(p=1))
    elif args.fold == -1:
        for fold in [0, 1, 2, 3, 4]:
            _, file_names = get_split(fold)
            model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=fold))),
                              model_type=args.model_type)

            print('num file_names = {}'.format(len(file_names)))

            output_path = Path(args.output_path)
            output_path.mkdir(exist_ok=True, parents=True)

            predict(model, file_names, args.batch_size, output_path, img_transform=img_transform(p=1))
    else:
        _, file_names = get_split(args.fold)
        model = get_model(str(Path(args.model_path).joinpath('model_{fold}.pt'.format(fold=args.fold))),
                          model_type=args.model_type)

        print('num file_names = {}'.format(len(file_names)))

        output_path = Path(args.output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        predict(model, file_names, args.batch_size, output_path, img_transform=img_transform(p=1))

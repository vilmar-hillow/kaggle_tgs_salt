import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from albumentations.torch.functional import img_to_tensor
import prepare_data


class SaltDataset(Dataset):
    def __init__(self, file_names: list, transform=None, mode='train'):
        self.file_names = file_names
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        if self.mode == 'train':
            mask = load_mask(img_file_name)
        else:
            mask = image

        data = {"image": image, "mask": mask}
        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        if self.mode == 'train':
            return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask_folder = 'masks'
    factor = prepare_data.binary_factor

    mask = cv2.imread(str(path).replace('images', mask_folder), 0)

    return (mask // factor).astype(np.uint8)

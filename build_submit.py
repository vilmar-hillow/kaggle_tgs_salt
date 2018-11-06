from pathlib import Path
import cv2
import numpy as np
import pandas as pd


pred_folder = Path('./predictions/')


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  # list of run lengths
    r = 0  # the current run length
    pos = 1  # count starts from 1 per WK
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


def read_mask(path):
    mask = cv2.imread(str(path), 0)

    return mask / 255


pred_dict = {}
folds = ["0", "1", "2", "3", "4"]
for fold in folds:
    folder = pred_folder / Path(fold)
    files = [f for f in folder.glob('*')]
    for fn in files:
        mask = read_mask(fn)
        if fold == "0":
            pred_dict[fn.stem] = mask
        else:
            pred_dict[fn.stem] += mask

final_dict = {}
for key, value in pred_dict.items():
    average = value / len(folds)
    mask = np.where(average > 0.5, 1, 0)
    final_dict[key] = RLenc(mask)

sub = pd.DataFrame.from_dict(final_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission_5_folds_resnext_0.5thr_300ep_256_01finetuned.csv')


from pathlib import Path
import argparse
import cv2
import numpy as np


def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--train_path', type=str, default='data/train',
        help='path where train images with ground truth are located')
    arg('--target_path', type=str, default='predictions', help='path with predictions')
    args = parser.parse_args()

    best_thr = None
    best_dice = 0

    for thr in np.arange(.3, 1., .01):
        threshold = thr
        result_dice = []
        result_jaccard = []

        for file_name in (Path(args.train_path) / 'masks').glob('*'):
            y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

            pred_file_name = Path(args.target_path) / file_name.name

            y_pred = (cv2.imread(str(pred_file_name), 0) > 255 * threshold).astype(np.uint8)

            result_dice += [dice(y_true, y_pred)]
            result_jaccard += [jaccard(y_true, y_pred)]
        
        mean_dice = np.mean(result_dice)
        if mean_dice > best_dice:
            best_thr = thr
            best_dice = mean_dice

        print('Threshold = {}'.format(round(threshold, 3)))
        print('Dice = ', mean_dice, np.std(result_dice))
        print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))
        print()
    print('Best threshold is {} with Dice score of {}'.format(round(best_thr, 3), round(best_dice, 5)))

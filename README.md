# Kaggle TGS Salt Identification Challenge solution

This is a quick and hacky solution that I managed to try during the last days of submission deadline.
This solution netted a 333rd place on the leaderboard (out of 3234 competitors).

Since I decided to work on the competition pretty late, I didn't manage to try a lot of encoders, losses and the like.
However, this shows that a blunt solution (some encoder with pretrained weights) works pretty well and can serve 
as a good start to make your way into a competition like this.

## Model

Out of the few encoders I've tried (ResNet, ResNext, WiderResnet), [se_resnext50_32x4d](https://github.com/Cadene/pretrained-models.pytorch#senet)
performed best, but heavier encoders work even better as evident from the top competitors' reports.

I couldn't make [lovasz loss](https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py)
that was used by many work and didn't have time to debug it, sadly.

## Dataset

The data was split into 5 folds, stratified by depth info.
Intitially I padded the input from 101 to 128, but resizing it to 202 and padding to 256 worked even better.

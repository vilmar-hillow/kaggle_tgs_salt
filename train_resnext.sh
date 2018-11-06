#!/bin/bash

for i in 0 1 2 3 4
do
   CUDA_VISIBLE_DEVICES=2 python train.py --batch-size 12 --fold $i --workers 8 --lr 0.001 --root 'runs/resnext' --model 'ResNext' --n-epochs 300 --loss StableBCE --jaccard-weight 0.5
   CUDA_VISIBLE_DEVICES=2 python train.py --batch-size 12 --fold $i --workers 8 --lr 0.0001 --root 'runs/resnext' --model 'ResNext' --n-epochs 600 --loss StableBCE --jaccard-weight 0.5
done

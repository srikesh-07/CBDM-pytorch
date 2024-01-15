#!/bin/bash

python classifier.py --arch resnet32 --dataset cifar10 --exp-name 'CIFAR10' --save-dir 'CIFAR10'
python classifier.py --arch resnet32 --dataset cifar10lt --imb_factor 0.01 --exp-name 'CIFAR10LT-0.01' --save-dir 'CIFAR10LT_0.01'
python classifier.py --arch resnet32 --dataset cifar10lt --imb_factor 0.05 --exp-name 'CIFAR10LT-0.05' --save-dir 'CIFAR10LT_0.05'

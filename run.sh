#!/bin/bash

python classifier.py --exp-name CIFAR10LT_0.0.005 --imb_factor 0.005
python classifier.py --exp-name CIFAR10LT_0.01 --imb_factor 0.01
python classifier.py --exp-name CIFAR10LT_0.02 --imb_factor 0.02
python classifier.py --exp-name CIFAR10LT_0.05 --imb_factor 0.05
python classifier.py --exp-name CIFAR10LT_0.1 --imb_factor 0.1
python classifier.py --exp-name CIFAR10LT_0.2 --imb_factor 0.2

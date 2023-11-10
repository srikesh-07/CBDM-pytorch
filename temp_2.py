from dataset import ImbalanceCIFAR100, ImbalanceCIFAR10
import json
import os
import math

if __name__ == '__main__':
    json_data = dict()

    cifar10_lt = ImbalanceCIFAR10('./', download=True)
    data = cifar10_lt.get_data_distribution()
    split = math.floor(len(data['classes']) / 3)
    classes = list(data['data_dist'].keys())
    data['dist_split'] = {"head": classes[:split],
                          'med': classes[split: split * 2],
                          'tail': classes[split * 2:]}
    json_data['CIFAR10_lt'] = data

    cifar100_lt = ImbalanceCIFAR100('./', download=True)
    data = cifar100_lt.get_data_distribution()
    split = math.floor(len(data['classes']) / 3)
    classes = list(data['data_dist'].keys())
    data['dist_split'] = {"head": classes[:split],
                          'med': classes[split: split * 2],
                          'tail': classes[split * 2:]}
    json_data['CIFAR100_lt'] = data

    with open('lt_distribution.json', 'w') as j_file:
        json.dump(json_data, j_file, indent=4)

    print("Successfully generated Data distribution config JSON files.")
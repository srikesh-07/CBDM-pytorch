import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset
from score.fid import get_statistics
import os
import json
from torchvision import transforms


tran_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize([32, 32])
])


def get_classwise_indices(dataset):
    indices = dict()
    for idx in range(len(dataset)):
        _, cls_id = dataset[idx]
        if indices.get(int(cls_id), None) is None:
            indices[int(cls_id)] = [idx]
        else:
            indices[int(cls_id)].append(idx)
    return indices


def check_classes(cls_1, cls_2):
    assert len(cls_1) == len(cls_2), "Invalid number of classes"
    for c1, c2 in zip(cls_1, cls_2):
        assert c1 == c2, "Invalid arrangement of classses"


def generate__npz(dataset, dist_dict, cls_indices):
    os.makedirs("custom_statistics", exist_ok=True)
    for split_name, class_ids in dist_dict['dist_split'].items():
        print(f"Generating Statistics for {split_name} in {dataset.__class__.__name__}")
        sampled_indices = list()
        for cls_id in class_ids:
            sampled_indices.extend(cls_indices[cls_id])
        print(len(sampled_indices))
        assert len(sampled_indices) == len(class_ids) * 5000
        imgs = list()
        for idx in sampled_indices:
            img, _ = dataset[idx]
            imgs.append(np.array(img))
        # imgs = np.stack(imgs, axis=0)
        # print(imgs.shape)
        mu, sigma = get_statistics(imgs)
        np.savez(os.path.join("custom_statistics", f"{dataset.__class__.__name__}_{split_name}"),
                 mu=mu,
                 sigma=sigma)


if __name__ == '__main__':

    with open('lt_distribution.json', 'r') as json_file:
        j_data = json.load(json_file)

    cifar10 = CIFAR10(root='./', train=True, download=True, transform=tran_transform)
    cifar10_indices = get_classwise_indices(cifar10)
    cifar10_dist = j_data['CIFAR10_lt']
    check_classes(cifar10.classes, cifar10_dist['classes'])
    generate__npz(cifar10, cifar10_dist, cifar10_indices)







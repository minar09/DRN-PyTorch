import argparse
import json
import numpy as np
from PIL import Image
from os import path as osp


def compute_mean_std():
    data_dir = 'E:/Dataset/Dataset10k/images/training/'
    list_dir = 'E:/Dataset/Dataset10k/list/'
    image_list_path = osp.join(list_dir, 'train.txt')
    image_list = [line.strip() for line in open(image_list_path, 'r')]
    np.random.shuffle(image_list)
    pixels = []
    for image_path in image_list[:500]:
        image = Image.open(osp.join(data_dir, image_path), 'r')
        pixels.append(np.asarray(image).reshape(-1, 3))
    pixels = np.vstack(pixels)
    mean = np.mean(pixels, axis=0) / 255
    std = np.std(pixels, axis=0) / 255
    print(mean, std)
    info = {'mean': mean.tolist(), 'std': std.tolist()}
    with open('info.json', 'w') as fp:
        json.dump(info, fp)


def main():
    compute_mean_std()


if __name__ == '__main__':
    main()

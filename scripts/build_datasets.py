# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from data.build_dataset import decom_BDD100KLabels
from data.data_augmentation import random_aimage_aug
from utils.utils import get_obj_from_str


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', choices=['ECP', 'CityPersons', 'BDD100K'], default='BDD100K')
    parser.add_argument('--base_dir', type=str, default=r'D:\my_phd\dataset\D4_BDD100K\bdd100k')
    parser.add_argument('--json_path', type=str, default=None)
    parser.add_argument('--crop_num', type=int, default=3000)

    opts = parser.parse_args()

    return opts


opts = get_opts()

ds_func_dict = {
    'ECP': 'data.build_dataset.Read_ECP',
    'CityPersons': 'data.build_dataset.Read_CityPersons',
    'BDD100K': 'data.build_dataset.Read_BDD100K',
}
set_list = ['train', 'val', 'train_extra'] if opts.dataset_name == 'CityPersons' else ['train', 'val']
type_list = ['pedestrian', 'nonPedestrian']

if opts.dataset_name == 'BDD100K':
    decom_BDD100KLabels(opts.json_path)

cur_ds = get_obj_from_str(ds_func_dict[opts.dataset_name])(opts.base_dir)

# Get crop.txt
for set_name in set_list:
    # cur_ds.get_crop_txt(set_name=set_name)
    cur_ds.get_num(set_name=set_name)

# Cropping according to the num
for type in type_list:
    cur_ds.crop_objects(set_list=set_list, type=type, crop_num=opts.crop_num)

print(f'Now spliting dataset into train, val and test set.')
cur_ds.split_dataset()

# Augmentating training set
rand_aug = random_aimage_aug(base_dir=opts.base_dir)
rand_aug()
rand_aug.gen_aug_txt()


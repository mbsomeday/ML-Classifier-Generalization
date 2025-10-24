# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from experiments.pedestrian_classification import Ped_Classifier


def get_args():
    parser = argparse.ArgumentParser()

    # model & data
    parser.add_argument('--ped_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--data_key', type=str, default='Stage6_org')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--train_txt', type=str, default='augmentation_train.txt')

    # train
    parser.add_argument('-base_lr', type=float, default=0.001)
    parser.add_argument('--isTrain', action='store_true', default=True)
    parser.add_argument('--min_train_epoch', type=int, default=15)
    parser.add_argument('--max_train_epoch', type=int, default=50)

    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--monitored_metric', type=str, default='loss')

    parser.add_argument('--rand_seed', type=int, default=13)

    args = parser.parse_args()

    return args


args = get_args()

seed_list = [90, 8, 13, 20, 73]


for cur_seed in seed_list:
    args.rand_seed = cur_seed

    ped_model = Ped_Classifier(args)
    ped_model.train()

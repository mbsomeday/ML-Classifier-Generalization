# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from experiments.hyperparam_searcg import HPSelection


def get_args():
    parser = argparse.ArgumentParser()

    # model & data
    parser.add_argument('--ped_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--data_key', type=str, default='Stage6_org')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_txt', type=str, default='train.txt')

    # train
    parser.add_argument('--min_train_epoch', type=int, default=3)
    parser.add_argument('--hp_dir', type=str, default='D:\my_phd\on_git\ML-Classifier-Generalization\HPcomb')

    # test
    parser.add_argument('--model_weights', type=str, default=None)

    parser.add_argument('--rand_seed', type=int, default=82)
    parser.add_argument('--isTrain', action='store_true')

    args = parser.parse_args()

    return args


args = get_args()
hp = HPSelection(args)

if args.isTrain:
    print(f'Current mode: Train')
    hp.hp_search()
else:
    print(f'Current mode: Test')
    hp.test()














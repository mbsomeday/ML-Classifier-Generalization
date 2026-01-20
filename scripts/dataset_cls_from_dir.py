# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from experiments.dataset_classification import ds_cls_from_dir

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_path', type=str, default=r'E:\Bias_Reduction_Summary\Datasets\Perturbations\D1_perturb')
    parser.add_argument('--ds_label', type=int, default=0)

    parser.add_argument('--ds_model_obj', default='torchvision.models.efficientnet_b0'),

    # test
    parser.add_argument('--ds_weights_path',
                        default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    parser.add_argument('--test_batch_size', type=int, default=2)

    opts = parser.parse_args()

    return opts


opts = get_opts()
ds_cls_from_dir(opts)
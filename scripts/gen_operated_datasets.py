import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, torch, random

from experiment_classes.operated_datasets import gen_perturbation_image


def get_opts():
    parser = argparse.ArgumentParser()

    # # 用于云端
    # parser.add_argument('--task', type=str, choices=['onlyPerturb', 'combined'], required=True, default=None)
    # parser.add_argument('--ds_name_list', nargs='+')
    # parser.add_argument('--txt_name', type=str)
    # parser.add_argument('--num_classes', type=int,
    #                     help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    # parser.add_argument('--model_weights', type=str)
    #
    # parser.add_argument('--path_key', type=str, default='Stage6_org')
    # parser.add_argument('--batch_size', type=int, default=1)
    #
    # # 生成perturbation图片
    # parser.add_argument('--perturb_save_dir', type=str, required=False)
    #
    # # 生成perturbation+aug图片
    # parser.add_argument('--perturb_dir', type=str, required=False)
    # parser.add_argument('--compondImg_save_dir', type=str, required=False)

    # 用于本地测试
    parser.add_argument('--task', type=str, choices=['perturbed', 'cmbined'], default='cmbined')
    parser.add_argument('--ds_name_list', nargs='+', default=['D2'])
    # parser.add_argument('--txt_name', type=str, default='test.txt')
    parser.add_argument('--txt_name_list', nargs='+', default=['val.txt', 'test.txt'])
    parser.add_argument('--num_classes', type=int, default=3, help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')

    # parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\baselines\D2\efficientNetB0_D2_51_Baseline-19-2.00064.pth')

    # parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--batch_size', type=int, default=1)

    # 生成perturbation图片
    parser.add_argument('--perturb_save_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\All_Processor\M3CAM\OnlyPerturb')

    # 生成perturbation+aug图片
    parser.add_argument('--perturb_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_citypersons\All_Processor\DSCAM\onlyPurturbations\test')
    # parser.add_argument('--perturb_dir', type=str, default=None)
    parser.add_argument('--compondImg_save_dir', type=str, default=None)

    opts = parser.parse_args()

    return opts


if __name__ == '__main__':
    opts = get_opts()
    gen_perturbation_image(opts)



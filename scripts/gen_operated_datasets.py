import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, torch, random

from experiment_classes.operated_datasets import gen_perturbation_image, gen_CAM_Mask, FUNC_REGISTRY


def get_opts():
    parser = argparse.ArgumentParser()

    # 用于云端
    parser.add_argument('--gen_task_name_list', nargs='+', default=['CAM_Mask', 'onlyPerturb', 'Aug_Perturb', 'Aug_Org'])
    parser.add_argument('--ds_name_list', nargs='+')
    parser.add_argument('--txt_name_list', nargs='+', required=True)
    parser.add_argument('--num_classes', type=int, help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    parser.add_argument('--model_weights', type=str)

    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--batch_size', type=int, default=1)

    # 生成图片的保存dir
    parser.add_argument('--genImg_save_dir', type=str, required=False, help='base dir path, no need to add the txt name')


    # # 用于本地测试
    # # parser.add_argument('--gen_task_name_list', nargs='+', default=['Aug_Perturb'])
    # parser.add_argument('--gen_task_name_list', nargs='+', default=['CAM_Mask', 'OnlyPerturb', 'Aug_Perturb', 'Aug_Org'])
    # parser.add_argument('--ds_name_list', nargs='+', default=['D3'])
    # parser.add_argument('--txt_name_list', nargs='+', default=['val.txt'])
    # parser.add_argument('--num_classes', type=int, default=3, help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    # parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    #
    # # parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\baselines\D2\efficientNetB0_D2_51_Baseline-19-2.00064.pth')
    #
    # # parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    # parser.add_argument('--path_key', type=str, default='Stage6_org')
    # parser.add_argument('--batch_size', type=int, default=1)
    #
    # # 生成图片的保存dir
    # parser.add_argument('--genImg_save_dir', type=str, default=r'D:\my_phd\on_git\ML-Classifier-Generalization\test_images\test_gen', help='base dir path, no need to add the txt name')

    # # 生成perturbation+aug图片
    # parser.add_argument('--perturb_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_citypersons\All_Processor\DSCAM\onlyPurturbations\test')
    # # parser.add_argument('--perturb_dir', type=str, default=None)
    # parser.add_argument('--compondImg_save_dir', type=str, default=None)

    opts = parser.parse_args()

    return opts


if __name__ == '__main__':
    opts = get_opts()

    for t_name in opts.gen_task_name_list:
        task_func = FUNC_REGISTRY.get(t_name)
        if task_func is None:
            raise ValueError(f'Unknown task name:{t_name}')

        print('-' * 10, f'Current running task:{t_name}', '-' * 10)
        task_func(opts)


    # gen_perturbation_image(opts)
    # gen_CAM_mask(opts)
















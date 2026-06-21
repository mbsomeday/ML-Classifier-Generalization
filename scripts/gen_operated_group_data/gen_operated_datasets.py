# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from experiment_classes.operated_datasets import GEN_FUNC_REGISTRY


import argparse
from datetime import datetime



def get_opts():
    parser = argparse.ArgumentParser()

    # 用于云端
    parser.add_argument('--model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--gen_task_name_list', nargs='+')
    parser.add_argument('--ds_name_list', nargs='+')
    parser.add_argument('--txt_name_list', nargs='+', required=True)
    parser.add_argument('--num_classes', type=int, help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    parser.add_argument('--model_weights', type=str)

    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dir_exist_ok', type=bool, default=False)


    # 生成图片的保存dir
    parser.add_argument('--genImg_save_dir', type=str, required=False, help='base dir path, no need to add the txt name')


    # # 用于本地测试
    # parser.add_argument('--model_obj', type=str, default='torchvision.models.efficientnet_b0')
    # parser.add_argument('--gen_task_name_list', nargs='+', default=['CAM_Mask'])
    # # parser.add_argument('--gen_task_name_list', nargs='+', default=['CAM_Mask', 'OnlyPerturb', 'Aug_Perturb', 'Aug_Org'])
    # parser.add_argument('--ds_name_list', nargs='+', default=['D3'])
    # parser.add_argument('--txt_name_list', nargs='+', default=['val.txt'])
    # parser.add_argument('--num_classes', type=int, default=2, help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    # parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\baselines\D3\efficientNetB0_D3_90_baseline\efficientNetB0_D3_90_Baseline-10-5.70169.pth')
    # parser.add_argument('--dir_exist_ok', type=bool, default=True)
    #
    # # parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    #
    # parser.add_argument('--path_key', type=str, default='Stage6_org')
    # parser.add_argument('--batch_size', type=int, default=1)
    #
    # # 生成图片的保存dir
    # parser.add_argument('--genImg_save_dir', type=str, default=r'D:\my_phd\on_git\ML-Classifier-Generalization\test_images\test_gen', help='base dir path, no need to add the txt name')


    opts = parser.parse_args()

    return opts


if __name__ == '__main__':
    opts = get_opts()

    start_time = datetime.now()
    print(f'Task starts at: {start_time}')

    for t_name in opts.gen_task_name_list:
        task_func = GEN_FUNC_REGISTRY.get(t_name)
        if task_func is None:
            raise ValueError(f'Unknown task name:{t_name}')

        print('-' * 10, f'Current running task:{t_name}', '-' * 10)
        task_func(opts)

    end_time = datetime.now()
    print(f'Task ends at:{end_time}\nDuration:{end_time-start_time}')














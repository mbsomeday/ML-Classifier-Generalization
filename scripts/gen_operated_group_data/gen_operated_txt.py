import argparse

from experiment_classes.operated_datasets import gen_txt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--add_org_txt', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\dataset_txt\train.txt', help='path to the org txt file.')
    parser.add_argument('--image_base_dir', type=str, default=r'E:\Bias_Reduction_Summary\Datasets')
    parser.add_argument('--txt_save_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_bdd100k\dataset_txt\exp_group')

    parser.add_argument('--imgae_group_path', type=str, default=r'Processor\D3_M3CAM\OrgAug\train')
    parser.add_argument('--txt_save_name', type=str, default='Org_M3CAMOrgAug.txt')

    opts = parser.parse_args()
    return opts


opts = get_args()
print(f'参数：{opts}')
gen_txt(opts)
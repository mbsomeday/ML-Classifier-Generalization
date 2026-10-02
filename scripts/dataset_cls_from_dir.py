'''
    本部分代码用dataset classifier直接测试dir_path下的图片属于哪个数据集
    ds_path结构：
        pedestrian
        nonPedestrian
'''


# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse

from experiment_classes.dataset_classification import ds_cls_from_dirs

def get_opts():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--ds_dir_list', nargs='+', default=[r'E:\Bias_Reduction_Summary\Datasets\Perturbations\D1_perturb'])
    # parser.add_argument('--ds_label_list', nargs='+', default=[0])
    #
    #
    # parser.add_argument('--ds_model_obj', default='torchvision.models.efficientnet_b0'),
    #
    # # test
    # parser.add_argument('--ds_weights_path', default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    # parser.add_argument('--test_batch_size', type=int, default=2)
    # parser.add_argument('--save_plt', action='store_true', help='not save CM by default')
    # parser.add_argument('--cm_save_dir', type=str, default=None)
    # parser.add_argument('--cm_title', type=str, default=None)

    parser.add_argument('--ds_dir_list', nargs='+', default=[r'D:\my_phd\dataset\Stage6\1001Temp\D1\Test', r'D:\my_phd\dataset\Stage6\1001Temp\D2\Test', R'D:\my_phd\dataset\Stage6\1001Temp\D3\Test'])
    parser.add_argument('--ds_label_list', nargs='+', default=[0, 1, 2])

    # parser.add_argument('--ds_model_obj', default='torchvision.models.efficientnet_b0')
    parser.add_argument('--ds_model_obj', default='torchvision.models.resnet18')


    # test
    parser.add_argument('--ds_weights_path', type=str, default=r'E:\Bias_Reduction_Summary\Backbone_ResNet18\Baselines\Dataset_Classifier\On_TrainSet\dsCls_D1D2D3_012\dsCls_D1D2D3_012-24-0.0365.pth')
    # parser.add_argument('--ds_weights_path', type=str, default=r'E:\Bias_Reduction_Summary\Backbone_ResNet18\Baselines\Dataset_Classifier\dsCls_D1D2D3_17\dsCls_D1D2D3_012-17-0.0315.pth')

    parser.add_argument('--test_batch_size', type=int, default=2)
    parser.add_argument('--save_plt', action='store_true', help='not save CM by default')
    parser.add_argument('--cm_save_dir', type=str, default=None)
    parser.add_argument('--cm_title', type=str, default=None)

    opts = parser.parse_args()

    return opts


opts = get_opts()
ds_cls_from_dirs(opts)
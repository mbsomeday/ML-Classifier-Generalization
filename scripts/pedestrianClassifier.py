'''
    本代码用于训练/测试 pedestrian classifier
    训练和测试需传入不同的参数

'''

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, random, time
import datetime

from experiment_classes.pedestrian_classification import Ped_Classifier


def get_args():
    parser = argparse.ArgumentParser()

    # 云端
    # model & data
    parser.add_argument('--ped_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'], help='the list means training_func on all of these datasets')
    parser.add_argument('--data_key', type=str, default='Stage6_org')
    parser.add_argument('--train_batch_size', type=int, default=64)   # 将train, val和test的batch size分开，方便loss的计算
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--train_txt', type=str, default='augmentation_train.txt')

    # train
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--min_train_epoch', type=int, default=10)
    parser.add_argument('--max_train_epoch', type=int, default=200)
    parser.add_argument('--seed_num', type=int, default=1, help='set the number of training_func times for getting the average value')

    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--monitored_metric', type=str, default='loss')
    parser.add_argument('--exp_name', type=str, default='', help='the name of the experiment, used for the dir name')

    # train with perturb+aug imagse
    parser.add_argument('--perturb_dir', type=str, default='', help='A temporal variable, for loading perturbed data')

    # test
    parser.add_argument('--ped_weights_path', type=str, default=None)
    parser.add_argument('--test_ds_list', nargs='+', default=None)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--save_plt', action='store_true', help='not save CM by default')
    parser.add_argument('--cm_save_dir', type=str, default=None)
    parser.add_argument('--cm_title', type=str, default=None)

    # # 本地
    # # model & data
    # parser.add_argument('--ped_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    # parser.add_argument('--ds_name_list', nargs='+', default=['D3'], help='the list means training_func on all of these datasets')
    # parser.add_argument('--data_key', type=str, default='Stage6_org')
    # parser.add_argument('--train_batch_size', type=int, default=4)   # 将train, val和test的batch size分开，方便loss的计算
    # parser.add_argument('--val_batch_size', type=int, default=4)
    # parser.add_argument('--train_txt', type=str, default=r'exp_group\Org_DSCAMOnlyPerturb.txt')
    #
    # # train
    # parser.add_argument('--base_lr', type=float, default=0.01)
    # parser.add_argument('--isTrain', action='store_true', default=True)
    # parser.add_argument('--min_train_epoch', type=int, default=10)
    # parser.add_argument('--max_train_epoch', type=int, default=200)
    # parser.add_argument('--seed_num', type=int, default=1, help='set the number of training_func times for getting the average value')
    #
    # parser.add_argument('--top_k', type=int, default=1)
    # parser.add_argument('--patience', type=int, default=10)
    # parser.add_argument('--warmup_epochs', type=int, default=3)
    # parser.add_argument('--monitored_metric', type=str, default='loss')
    # parser.add_argument('--exp_name', type=str, default='', help='the name of the experiment, used for the dir name')
    #
    # # train with perturb+aug imagse
    # parser.add_argument('--perturb_dir', type=str, default='', help='A temporal variable, for loading perturbed data')
    #
    # # test
    # parser.add_argument('--ped_weights_path', type=str, default=None)
    # parser.add_argument('--test_ds_list', nargs='+', default=None)
    # parser.add_argument('--test_batch_size', type=int, default=128)
    # parser.add_argument('--save_plt', action='store_true', help='not save CM by default')
    # parser.add_argument('--cm_save_dir', type=str, default=None)
    # parser.add_argument('--cm_title', type=str, default=None)

    args = parser.parse_args()

    return args


args = get_args()

# 开始时间
start_time = datetime.datetime.now()
print(f'Started at {str(start_time.strftime("%Y-%m-%d %H:%M:%S"))}')

if args.isTrain:
    print('Current Mode: 【Training】')
    # 生成程度为n的seed_list
    seed_list = random.sample(range(0, 100), args.seed_num)
    for cur_seed in seed_list:
        # 先调节seed再创建ped实例
        setattr(args, 'rand_seed', cur_seed)    # 向args中添加rand seed

        ped_model = Ped_Classifier(args)
        ped_model.train()
else:
    print('Current Mode: 【Testing】')
    ped_model = Ped_Classifier(args)
    ped_model.test()


# 结束时间
end_time = datetime.datetime.now()
duration = end_time - start_time
print(f'Ended at {str(end_time.strftime("%Y-%m-%d %H:%M:%S"))}')
print(f'Duration: {str(duration)}')
















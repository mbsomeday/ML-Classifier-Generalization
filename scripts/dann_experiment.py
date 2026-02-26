# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)


import argparse, random, torch, os, datetime
import numpy as np


from experiments.dann_train import DANN_Trainer


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--source', nargs='+', default=['D1'])
    parser.add_argument('--target', nargs='+', default=['D2'])
    parser.add_argument('--path_key', type=str, default='Stage6_org')
    parser.add_argument('--train_txt', type=str, default='train.txt')       # augmentation_train
    parser.add_argument('--val_txt', type=str, default='val.txt')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)


    # train
    parser.add_argument('--monitored_metric', default='loss')
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--seed_list', nargs='+', default=[82])

    # callbacks
    parser.add_argument('--top_k', default=1)
    parser.add_argument('--patience', default=5)

    # test
    parser.add_argument('--test_ds_list', nargs='+', default=None)
    parser.add_argument('--weight_dir', type=str, default='./model')
    parser.add_argument('--test_txt', type=str, default=None, help='txt file that records test results')

    args = parser.parse_args()

    return args


args = get_args()

# 开始时间
start_time = datetime.datetime.now()
print(f'Started at {str(start_time.strftime("%Y-%m-%d %H:%M:%S"))}')

if args.isTrain:
    print('Current Mode: 【Training】')
    # 遍历每个seed
    for cur_seed in args.seed_list:
        setattr(args, 'cur_seed', cur_seed)
        dann_cls = DANN_Trainer(args)

else:
    print('Current Mode: 【Testing】')


# 结束时间
end_time = datetime.datetime.now()
duration = end_time - start_time
print(f'Ended at {str(end_time.strftime("%Y-%m-%d %H:%M:%S"))}')
print(f'Duration: {str(duration)}')


# manual_seed = args.seed
# random.seed(manual_seed)
# torch.manual_seed(manual_seed)
# np.random.seed(manual_seed)
# os.environ['PYTHONHASHSEED'] = str(manual_seed)
#
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(manual_seed)         # 设置当前GPU的seed
#     torch.cuda.manual_seed_all(manual_seed)     # 有多个GPU的情况，确保所有GPU都用相同的seed
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
# start_time = datetime.datetime.now()
# print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))
#
# dann_cls = DANN_Trainer(args)
# if args.isTrain:
#     dann_cls.train()
#     end_time = datetime.datetime.now()
#     duration = end_time - start_time
#     print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
#     print("Duration: " + str(duration))
# else:
#     dann_cls.test()














































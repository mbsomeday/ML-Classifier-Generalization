# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, datetime

from experiments.dataset_classification import DS_Classifier

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--num_cls', type=int, default=3)
    parser.add_argument('--ds_name_list', nargs='+', default=['D1', 'D2', 'D3'])
    parser.add_argument('--data_key', default='Stage6_org')
    parser.add_argument('--ds_labels', nargs='+', default=['0', '1', '2'])

    # train
    parser.add_argument('--isTrain', action='store_true')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--monitored_metric', type=str, default='loss')
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--min_epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=3)

    # val
    parser.add_argument('--val_batch_size', default=64)

    # test
    parser.add_argument('--ds_weights_path', type=str, default=None)
    parser.add_argument('--test_txt_name', default='test.txt')
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--cm_save_dir', type=str, default=None)
    parser.add_argument('--cm_title', type=str, default=None)

    # callback
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)

    opts = parser.parse_args()

    return opts


opts = get_opts()
ds_cls = DS_Classifier(opts)

# 开始时间
start_time = datetime.datetime.now()
print(f'Started at {str(start_time.strftime("%Y-%m-%d %H:%M:%S"))}')

if opts.isTrain:
    print('Current Mode: 【Training】')
    ds_cls.train()
else:
    print('Current Mode: 【Testing】')
    ds_cls.test()

# 结束时间
end_time = datetime.datetime.now()
duration = end_time - start_time
print(f'Ended at {str(end_time.strftime("%Y-%m-%d %H:%M:%S"))}')
print(f'Duration: {str(duration)}')



























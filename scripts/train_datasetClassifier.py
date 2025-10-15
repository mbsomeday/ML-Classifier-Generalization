import argparse

from experiments.dataset_classification import DS_Classifier

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--num_cls', type=int, default=3)
    parser.add_argument('--ds_name_list', nargs='+', default=['D3'])
    parser.add_argument('--data_key', default='Stage6_org')
    parser.add_argument('--ds_labels', nargs='+', default=['0'])

    # train
    parser.add_argument('--isTrain', default=False)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--monitored_metric', type=str, default='accuracy')
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--min_epochs', type=int, default=30)
    parser.add_argument('--warmup_epochs', type=int, default=1)

    # val
    parser.add_argument('--val_batch_size', default=4)

    # test
    parser.add_argument('--ds_weights_path', type=str, default=None)
    parser.add_argument('--test_txt_name', default='test.txt')
    parser.add_argument('--test_batch_size', default=2)

    # callback
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--patience', type=int, default=2)

    opts = parser.parse_args()

    return opts


opts = get_opts()
ds_cls = DS_Classifier(opts)
ds_cls.test()




























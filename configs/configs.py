import argparse


def trainPedestrian_args():
    parser = argparse.ArgumentParser()

    # model & data
    parser.add_argument('--ped_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--data_key', type=str, default='Stage6_org')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--train_txt', type=str, default='augmentation_train.txt')

    # train
    parser.add_argument('-base_lr', type=float, default=0.001)
    parser.add_argument('--isTrain', action='store_true', default=True)
    parser.add_argument('--min_train_epoch', type=int, default=30)
    parser.add_argument('--max_train_epoch', type=int, default=100)

    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--monitored_metric', type=str, default='loss')

    args = parser.parse_args()

    return args


def testPedestrian_args():
    parser = argparse.ArgumentParser()

    # model & data
    parser.add_argument('--ped_model_obj', type=str, default='torchvision.models.efficientnet_b0')
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--data_key', type=str, default='Stage6_org')
    parser.add_argument('--batch_size', type=int, default=4)

    # test
    parser.add_argument('--ped_weights_path', type=int, default=None)

    args = parser.parse_args()

    return args


































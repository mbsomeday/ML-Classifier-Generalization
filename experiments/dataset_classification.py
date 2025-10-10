import torch, os
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from data.dataset import my_dataset
from training.callbacks import EarlyStopping
from utils.utils import get_obj_from_str, DEVICE, DotDict, load_model


class DS_Classifier():
    '''
        class for training the dataset classifier
    '''
    def __init__(self, opts):
        # the experiment only requires one GPU
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                raise RuntimeError('More than one GPU')
            else:
                print(f'Runing on {torch.cuda.get_device_name(0)} GPU')

        self.opts = opts
        self.print_args()

        self.ds_model = get_obj_from_str(self.opts.ds_model_obj)(weights=None, progress=True, num_classes=opts.num_cls).to(DEVICE)

        if self.opts.isTrain:
            self.training_setup()

    def print_args(self):
        '''
            Printing args to the console & Saing args to .txt file

        '''
        print('-' * 40 + ' Args ' + '-' * 40)

        self.callback_dir = 'dsCls' + '_' + ''.join(self.opts.ds_name_list) + '_' + ''.join(self.opts.ds_labels)
        self.callback_path = os.path.join(os.getcwd(), self.callback_dir)
        if not os.path.exists(self.callback_path):
            os.mkdir(self.callback_path)
        print(f'Callback dir: {self.callback_path}')

        with open(os.path.join(self.callback_path, 'Args.txt'), 'a') as f:
            for k, v in vars(self.opts).items():
                msg = f'{k}: {v}'
                print(msg)
                f.write(msg + '\n')

    def training_setup(self):
        print('-' * 40 + ' Init Model & Loading Data ' + '-' * 40)

        # ********** 模型初始化 **********
        self.init_model(self.ds_model)

        # ********** 数据准备 **********    augmentation_train
        self.train_dataset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key=self.opts.data_key, txt_name='train.txt', ds_labels=self.opts.ds_labels)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opts.train_batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key=self.opts.data_key, txt_name='val.txt', ds_labels=self.opts.ds_labels)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.opts.val_batch_size, shuffle=False)

        # ********** loss & scheduler **********
        self.optimizer = torch.optim.RMSprop(self.ds_model.parameters(), lr=self.opts.base_lr, weight_decay=1e-5, eps=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # ********** callbacks **********
        self.start_epoch = 0
        self.early_stopping = EarlyStopping(self.callback_path, top_k=self.opts.top_k, cur_epoch=self.start_epoch, patience=self.opts.patience, monitored_metric=self.opts.monitored_metric)
    #
    def init_model(self, model):
        '''
            对模型权重进行正交初始化，保障多次训练结果变动不会变化太大
            适用 kaiming init，suitable for ReLU
            初始化种类参考： https://blog.csdn.net/shanglianlm/article/details/85165523
        '''
        print('Initilizing model weights using orthogonal.')
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  #  kaiming_uniform_
                nn.init.orthogonal_(m.weight)     # 正交初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        return model

    def update_learning_rate(self, epoch):
        '''
            warmup + learning rate decay
        :param epoch:
        :return:
        '''
        old_lr = self.optimizer.param_groups[0]['lr']

        # warm-up阶段
        if epoch <= self.opts.warmup_epochs:  # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.opts.base_lr * epoch / self.opts.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.opts.base_lr * 0.963 ** (epoch / 3)  # gamma=0.963, lr decay epochs=3

        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def train_one_epoch(self):
        self.ds_model.train()
        epoch_loss = 0.0
        correct_num = 0

        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            images = data['image'].to(DEVICE)
            ds_labels = data['ds_label'].to(DEVICE)

            logits = self.ds_model(images)
            pred = torch.argmax(logits, 1)
            loss_value = self.loss_fn(logits, ds_labels)

            epoch_loss += loss_value.item()
            correct_num += (pred == ds_labels).sum()

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

        train_accuracy = correct_num / len(self.train_dataset)
        print(f'Training loss:{epoch_loss:.6f}, accuracy:{train_accuracy:.4f}')


    def val_on_epoch_end(self):
        self.ds_model.eval()
        val_correct_num = 0.0
        val_loss = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.val_loader)):
                images = data['image'].to(DEVICE)
                ds_labels = data['ds_label'].to(DEVICE)

                logits = self.ds_model(images)
                preds = torch.argmax(logits, 1)
                loss_value = self.loss_fn(logits, ds_labels)

                val_loss += loss_value.item()
                val_correct_num += (preds == ds_labels).sum()

                y_true.extend(ds_labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_accuracy = val_correct_num / len(self.val_dataset)
        val_cm = confusion_matrix(y_true, y_pred)

        print(f'Val loss {val_loss:.6f}, accuracy:{val_accuracy:.4f}')
        print(f'CM on validation set:\n{val_cm}')

        val_epoch_info = {
            'accuracy': val_accuracy,
            'loss': val_loss
        }
        return DotDict(val_epoch_info)


    def train(self):
        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print('Total Batch:', len(self.train_loader))

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))
        for EPOCH in range(self.start_epoch, self.opts.max_epochs):
            print('=' * 30 + ' begin EPOCH ' + str(EPOCH + 1) + '=' * 30)
            self.train_one_epoch()
            self.update_learning_rate(EPOCH + 1)

            if EPOCH > self.opts.min_epochs:
                val_epoch_info = self.val_on_epoch_end()
                self.early_stopping(EPOCH + 1, self.ds_model, self.optimizer, val_epoch_info, scheduler=None)

            if self.early_stopping.early_stop:
                print(f'Early Stopping!')
                break

    def test(self):
        '''
            在 new D1/D2/D3 的test.txt上测试，展示结果但不保存
        '''
        self.ds_model = load_model(self.ds_model, self.opts.ds_weights_path)
        self.ds_model.eval()

        test_dataset = my_dataset(self.opts.ds_name_list, path_key=self.opts.data_key, txt_name=self.opts.test_txt_name, ds_labels=self.opts.ds_labels)
        test_loader = DataLoader(test_dataset, batch_size=self.opts.test_batch_size, shuffle=False)

        test_correct_num = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(test_loader)):
                images = data['image'].to(DEVICE)
                ds_labels = data['ds_label'].to(DEVICE)

                logits = self.ds_model(images)
                preds = torch.argmax(logits, 1)

                test_correct_num += (preds == ds_labels).sum()

                y_true.extend(ds_labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        test_accuracy = test_correct_num / len(test_dataset)
        test_cm = confusion_matrix(y_true, y_pred)
        print(f'Test accuracy:{test_accuracy:.6f}\nTest CM:{test_cm}')




if __name__ == '__main__':
    import argparse

    def get_opts():
        parser = argparse.ArgumentParser()

        parser.add_argument('--ds_model_obj', default='torchvision.models.efficientnet_b0'),
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
        parser.add_argument('--ds_weights_path', default='D:\my_phd\on_git\PerceptionBaseline\experiments\dsCls_D3_0\dsCls_D3_0-01-1.00000.pth')
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

























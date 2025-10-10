import torch, os
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tqdm import tqdm

from data.dataset import my_dataset
from utils.utils import DEVICE, get_obj_from_str, DotDict, load_model
from training.callbacks import EarlyStopping, Model_Logger



class Ped_Classifier():
    '''
        该类是在train的过程中对org image进行cam operation，
    '''
    def __init__(self, opts):

        # 确保在服务器运行时只用一个GPU
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                raise RuntimeError('More than one GPU')
            else:
                print(f'Runing on {torch.cuda.get_device_name(0)} GPU')

        self.opts = opts
        self.ped_model = get_obj_from_str(self.opts.ped_model_obj)(num_classes=2).to(DEVICE)

        if self.opts.isTrain:
            self.training_setup()
        else:
            # 若是测试，则创建 test 文件夹用于存储结果
            self.callback_save_path = os.path.join(os.getcwd(), 'Test')
            if not os.path.exists(self.callback_save_path):
                os.mkdir(self.callback_save_path)
            print(f'Test saving dir:{self.callback_save_path}')

        self.print_args()


    def training_setup(self):
        '''
            初始化训练的各种参数
        '''

        # ********** 创建 callback save dir **********
        # callback文件夹的模板为 model_{D1}_Baseline_{seed}
        self.callback_save_dir = self.opts.ped_model_obj.rsplit('.')[-1] + '_' + ''.join(self.opts.ds_name_list) + '_Baseline' + '_' + str(self.opts.rand_seed)
        self.callback_save_path = os.path.join(os.getcwd(), self.callback_save_dir)
        print(f'Callback_save_dir:{self.callback_save_path}')
        if not os.path.exists(self.callback_save_path):
            os.mkdir(self.callback_save_path)

        self.train_dataset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key=self.opts.data_key, txt_name=self.opts.train_txt)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opts.batch_size, shuffle=True)

        self.val_dataset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key=self.opts.data_key, txt_name='val.txt')
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.opts.batch_size, shuffle=False)

        self.train_nonPed_num, self.train_ped_num = self.train_dataset.get_ped_cls_num()
        self.val_nonPed_num, self.val_ped_num = self.val_dataset.get_ped_cls_num()

        # ********** loss & scheduler **********
        self.optimizer = torch.optim.RMSprop(self.ped_model.parameters(), lr=self.opts.base_lr, weight_decay=1e-5, eps=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # self.best_val_bc = -np.inf  # 监控balanced accuracy
        self.best_val_loss = np.inf # 监控loss
        self.ped_model = self.init_model(self.ped_model)

        # ********** callbacks **********
        self.early_stopping = EarlyStopping(self.callback_save_path, top_k=self.opts.top_k, cur_epoch=0, patience=self.opts.patience, monitored_metric=self.opts.monitored_metric)

        train_num_info = [len(self.train_dataset), self.train_nonPed_num, self.train_ped_num]
        val_num_info = [len(self.val_dataset), self.val_nonPed_num, self.val_ped_num]

        self.epoch_logger = Model_Logger(save_dir=self.callback_save_path,
                                         model_name=self.opts.ped_model_obj.split('.')[-1],
                                         ds_name_list=self.opts.ds_name_list,
                                         train_num_info=train_num_info,
                                         val_num_info=val_num_info,
                                         )

    def print_args(self):
        '''
            参数打印 并 保存到txt文件中
        '''
        print('-' * 40 + ' Args ' + '-' * 40)

        info = []
        for k, v in vars(self.opts).items():
            msg = f'{k}: {v}'
            print(msg)
            info.append(msg)

        # 将本次实验的参数写入txt中
        write_to_txt = os.path.join(self.callback_save_path, 'Args.txt')
        if os.path.exists(write_to_txt):
            os.remove(write_to_txt)
        with open(write_to_txt, 'a') as f:
            for item in info:
                f.write(item+'\n')

    def init_model(self, model):
        '''
            对模型权重进行初始化，保障多次训练结果变动不会变化太大
            适用 kaiming init，suitable for ReLU
            初始化种类参考： https://blog.csdn.net/shanglianlm/article/details/85165523
        '''
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # 或 kaiming_uniform_
                nn.init.orthogonal_(m.weight)     # 正交初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        return model

    def inif_pred_info(self):
        pred_info = {
            'y_true': [],
            'y_pred': [],
            'nonPed_acc_num': 0,
            'ped_acc_num': 0,
            'correct_num': 0,
            'loss': 0.0
        }
        return pred_info

    def handle_pred_info(self, org_pred: dict, info_type='Train'):
        '''
            整合训练过程中的 accuracy 和 loss 等数据并进行 输出 和 返回
        '''
        epoch_info = {}

        correct_num = org_pred['correct_num']
        accuracy = correct_num / len(org_pred['y_true'])
        balanced_accuracy = balanced_accuracy_score(org_pred['y_true'], org_pred['y_pred'])
        loss = org_pred['loss']

        epoch_info['accuracy'] = accuracy
        epoch_info['balanced_accuracy'] = balanced_accuracy
        epoch_info['loss'] = loss

        msg = f'Overall accuracy: {accuracy:.6f}, Overall balanced accuracy:{balanced_accuracy:.4f}, loss:{loss:.6f}'

        print('-' * 30, str(info_type) + ' Info' + '-' * 30)
        print(msg)

        return DotDict(epoch_info)

    def train_one_epoch(self):
        self.ped_model.train()

        # y_true = []
        org_dict = self.inif_pred_info()

        for batch_idx, data in enumerate(tqdm(self.train_loader)):
            images = data['image'].to(DEVICE)
            ped_labels = data['ped_label'].to(DEVICE)

            logits = self.ped_model(images)
            pred = torch.argmax(logits, 1)
            loss_value = self.loss_fn(logits, ped_labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            # ------------ 对 pred 进行记录 ------------
            # y_true.extend(ped_labels.cpu().numpy())
            nonPed_idx = (ped_labels == 0)
            ped_idx = (ped_labels == 1)

            org_dict['loss'] += loss_value.item()
            org_dict['y_true'].extend(ped_labels.cpu().numpy())
            org_dict['y_pred'].extend(pred.cpu().numpy())
            org_dict['correct_num'] += (pred == ped_labels).sum()
            org_dict['nonPed_acc_num'] += ((ped_labels[nonPed_idx] == pred[nonPed_idx]) * 1).sum()
            org_dict['ped_acc_num'] += ((ped_labels[ped_idx] == pred[ped_idx]) * 1).sum()



        train_epoch_info = self.handle_pred_info(org_pred=org_dict, info_type='Train')

        return train_epoch_info

    def val_on_epoch_end(self):
        self.ped_model.eval()

        y_true = []
        y_pred = []
        nonPed_acc_num = 0
        ped_acc_num = 0
        val_correct_num = 0
        val_loss = 0

        # self.val_nonPed_num, self.val_ped_num = self.val_dataset.get_ped_cls_num()

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.val_loader)):
                images = data['image'].to(DEVICE)
                ped_labels = data['ped_label'].to(DEVICE)

                logits = self.ped_model(images)
                preds = torch.argmax(logits, dim=1)
                val_loss += self.loss_fn(logits, ped_labels)

                y_true.extend(ped_labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                val_correct_num += (preds == ped_labels).sum()

                nonPed_idx = (ped_labels == 0)
                nonPed_acc_num += (ped_labels[nonPed_idx] == preds[nonPed_idx]).sum()
                ped_idx = (ped_labels == 1)
                ped_acc_num += ((ped_labels[ped_idx] == preds[ped_idx]) * 1).sum()



        val_accuracy = val_correct_num / len(self.val_dataset)
        val_bc = balanced_accuracy_score(y_true, y_pred)

        val_epoch_info = {
            'accuracy': val_accuracy,
            'balanced_accuracy': val_bc,
            'loss': val_loss
        }
        print(f'Validation accuracy:{val_accuracy:.6f}, balanced_accuracy:{val_bc:.6f}, loss:{val_loss:.8f}')

        return DotDict(val_epoch_info)

    def test(self):
        '''
            遍历每个数据集对模型进行测试，并将结果保存到Test文件夹中
        '''
        self.ped_model = load_model(self.ped_model, self.opts.ped_weights_path)
        self.ped_model.eval()

        write_to_txt = os.path.join(self.callback_save_path, 'Test.txt')
        with open(write_to_txt, 'a') as f:
            f.write('-' * 80 + '\n')
            f.write(f'Testing modle: {self.opts.ped_weights_path}.\n')
            f.write('ds_name, test_ba, tnr, tpr, tn, fp, fn, tp\n')

        for ds_name in self.opts.ds_name_list:
            test_dataset = my_dataset(ds_name_list=[ds_name], path_key=self.opts.data_key, txt_name_list=['test.txt'])
            test_loader = DataLoader(test_dataset, batch_size=self.opts.batch_size, shuffle=False)

            y_true = []
            y_pred = []
            nonPed_acc_num = 0
            ped_acc_num = 0
            test_correct_num = 0

            test_nonPed_num, test_ped_num = test_dataset.get_ped_cls_num()

            with torch.no_grad():
                for batch_idx, data in enumerate(tqdm(test_loader)):
                    images = data['image'].to(DEVICE)
                    ped_labels = data['ped_label'].to(DEVICE)

                    logits = self.ped_model(images)
                    preds = torch.argmax(logits, dim=1)

                    y_true.extend(ped_labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

                    test_correct_num += (preds == ped_labels).sum()

                    nonPed_idx = (ped_labels == 0)
                    nonPed_acc_num += (ped_labels[nonPed_idx] == preds[nonPed_idx]).sum()
                    ped_idx = (ped_labels == 1)
                    ped_acc_num += ((ped_labels[ped_idx] == preds[ped_idx]) * 1).sum()

            test_accuracy = test_correct_num / len(test_dataset)
            test_bc = balanced_accuracy_score(y_true, y_pred)

            test_nonPed_acc = nonPed_acc_num / test_nonPed_num
            test_ped_acc = ped_acc_num / test_ped_num

            test_cm = confusion_matrix(y_true, y_pred)

            print('-' * 40 + 'Test Info' + '-' * 40)
            msg = f'DS_name:{ds_name}, Balanced accuracy:{test_bc:.4f}, accuracy: {test_accuracy:.4f}\nNon-ped accuracy:{test_nonPed_acc:.4f}({nonPed_acc_num}/{test_nonPed_num})\nPed accuracy:{test_ped_acc:.4f}({ped_acc_num}/{test_ped_num})'
            print(msg)
            print(f'CM on test set:\n{test_cm}')

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print(tn, fp, fn, tp)

            with open(write_to_txt, 'a') as f:
                f.write(f'{ds_name}, {test_bc:.6f}, {test_nonPed_acc:.4f}, {test_ped_acc:.4f}, {tn}, {fp}, {fn}, {tp}\n')


    def update_learning_rate(self, epoch):
        old_lr = self.optimizer.param_groups[0]['lr']

        # warm-up阶段
        if epoch <= self.opts.warmup_epochs:  # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.opts.base_lr * epoch / self.opts.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.opts.base_lr * 0.963 ** (epoch / 3)  # gamma=0.963, lr decay epochs=3

        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))


    def train(self):
        print('-' * 20 + 'Training Info' + '-' * 20)
        print('Total training Samples:', len(self.train_dataset))
        print('Total Batch:', len(self.train_loader))

        print('-' * 20 + 'Validation Info' + '-' * 20)
        print('Total Val Samples:', len(self.val_dataset))

        for EPOCH in range(self.opts.max_train_epoch):

            print('=' * 30 + ' begin EPOCH ' + str(EPOCH + 1) + '=' * 30)
            train_epoch_info = self.train_one_epoch()
            val_epoch_info = self.val_on_epoch_end()

            # ------------------------ 调用callbacks ------------------------
            self.epoch_logger(epoch=EPOCH + 1, training_info=train_epoch_info, val_info=val_epoch_info)

            # ------------------------ 调用callbacks ------------------------
            # 每个epoch end调整learning rate
            self.update_learning_rate(EPOCH)

            # 当训练次数超过最低epoch时，其中early_stop策略
            if (EPOCH + 1) > self.opts.min_train_epoch:
                self.early_stopping(EPOCH + 1, self.ped_model, self.optimizer, val_epoch_info, scheduler=None)

                if self.early_stopping.early_stop:
                    print(f'Early Stopping!')
                    break


























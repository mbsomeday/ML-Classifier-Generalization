'''
    To find the best combination of hyperparameters
'''
import os, re
from torch.utils.data import random_split, DataLoader
from torch import nn
import itertools, torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from training.callbacks import EarlyStopping

from data.dataset import my_dataset
from utils.utils import DEVICE, load_model


# 把每个 combination 单独存放到自己的txt中

class HPSelection():
    def __init__(self, opts):
        super().__init__()

        hp_dict = {
            # 'batch_size': [48, 64, 128],
            'batch_size': [2, 4],
            'base_lr': [1e-3, 5e-4],
            'optimizer': ['Adam', 'SGD'],
            'scheduler': ['COS']
            }
        self.all_combinations = list(itertools.product(
            hp_dict['batch_size'],
            hp_dict['base_lr'],
            hp_dict['optimizer'],
            hp_dict['scheduler']
        ))

        self.opts = opts

        # 固定的hyperparamters
        self.warmup_epochs = 3
        self.min_epochs = 10
        self.max_epochs = 30
        self.mini_train_num = 500    # 500
        self.mini_val_num = 500      # 500
        self.mini_test_num = 500

        if self.opts.isTrain:
            self.train_setup()


    def train_setup(self):
        get_trainset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key='Stage6_org', txt_name='train.txt')
        self.mini_trainset, _ = random_split(get_trainset, [self.mini_train_num, len(get_trainset) - self.mini_train_num])
        print(f'MiniTrainset samples: {len(self.mini_trainset)}')

        get_valset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key='Stage6_org', txt_name='val.txt')
        self.mini_valset, _ = random_split(get_valset, [self.mini_val_num, len(get_valset) - self.mini_val_num])
        self.mini_valloader = DataLoader(self.mini_valset, batch_size=64, shuffle=False)
        print(f'MiniValset samples: {len(self.mini_valset)}')

        self.ped_model = models.efficientnet_b0(weights=None, num_classes=2)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # callbacks
        self.callback_save_dir = self.opts.ped_model_obj.rsplit('.')[-1] + '_' + ''.join(self.opts.ds_name_list) + '_Baseline' + '_' + str(self.opts.rand_seed)
        self.callback_save_path = os.path.join(self.opts.hp_dir, self.callback_save_dir)
        print(f'Callback_save_dir:{self.callback_save_path}')
        if not os.path.exists(self.callback_save_path):
            os.makedirs(self.callback_save_path)

        self.early_stopping = EarlyStopping(callback_path=self.callback_save_path, patience=self.warmup_epochs)
        self.txt_dir = os.path.join(self.opts.hp_dir, 'hp_txt')
        os.makedirs(self.txt_dir, exist_ok=True)

        print('共有的超参数组合数：', len(self.all_combinations))


    def val_on_epoch_end(self, epoch):

        val_info = {
            'loss': 0.0
        }

        self.ped_model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.mini_valloader, desc=f'Epoch {epoch} val')):
                images = data['image'].to(DEVICE)
                ped_labels = data['ped_label'].to(DEVICE)

                logits = self.ped_model(images)
                pred = torch.argmax(logits, 1)
                loss_value = self.loss_fn(logits, ped_labels)

                y_true.extend(ped_labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

                val_info['loss'] += loss_value.item()

            balanced_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
            val_info['balanced_accuracy'] = balanced_accuracy

        return val_info

    def test(self):
        '''
            最终在test set上进行检验
        '''

        # load data
        get_testset = my_dataset(ds_name_list=self.opts.ds_name_list, path_key='Stage6_org', txt_name='test.txt')
        self.mini_testset, _ = random_split(get_testset, [self.mini_val_num, len(get_testset) - self.mini_test_num])
        self.mini_testloader = DataLoader(self.mini_testset, batch_size=64, shuffle=False)
        print(f'MiniTestset samples: {len(self.mini_testset)}')

        # load model weights
        self.ped_model = load_model(model=self.ped_model, weights_path=self.opts.model_weights)
        self.ped_model.eval()

        y_true = []
        y_pred = []
        test_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.mini_testloader, desc='test')):
                images = data['image'].to(DEVICE)
                ped_labels = data['ped_label'].to(DEVICE)

                logits = self.ped_model(images)
                pred = torch.argmax(logits, 1)
                loss_value = self.loss_fn(logits, ped_labels)

                y_true.extend(ped_labels.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

                test_loss += loss_value.item()

            balanced_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
            print(f'Test loss: {test_loss}, test balanced acc: {balanced_accuracy}')

    def train_one_epoch(self, epoch):
        train_info = {
            'loss': 0.0,
        }

        self.ped_model.train()

        y_true = []
        y_pred = []
        for batch_idx, data in enumerate(tqdm(self.mini_trainloader, desc=f'Epoch {epoch} train')):
            images = data['image'].to(DEVICE)
            ped_labels = data['ped_label'].to(DEVICE)

            logits = self.ped_model(images)
            pred = torch.argmax(logits, 1)
            loss_value = self.loss_fn(logits, ped_labels)

            # 反向传播
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            y_true.extend(ped_labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            train_info['loss'] += loss_value.item()

        balanced_accuracy = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        train_info.update({'balanced_accuracy': balanced_accuracy})

        return train_info


    def hp_search(self):
        for idx, comb_info in enumerate(self.all_combinations):
            self.batch_size, self.base_lr, optimizer_type, scheduler_type = comb_info
            comb = [str(self.batch_size), str(self.base_lr), optimizer_type, scheduler_type]
            comb_name = '_'.join(comb)
            cur_txt_path = os.path.join(self.txt_dir, str(idx+1)+'.txt')
            print(f'comb_name:{comb_name}, cur_txt_path:{cur_txt_path}')
            with open(cur_txt_path, 'a') as f:
                f.write('Combination: ' + comb_name + '\n')

            self.mini_trainloader = DataLoader(self.mini_trainset, batch_size=self.batch_size, shuffle=True)

            if optimizer_type == 'Adam':
                self.optimizer = Adam(params=self.ped_model.parameters(), lr=self.base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            elif optimizer_type == 'SGD':
                self.optimizer = SGD(self.ped_model.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=0.0001)

            if scheduler_type == 'COS':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs-self.warmup_epochs)
            elif scheduler_type == 'EXP':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

            for EPOCH in range(self.max_epochs):
                print('=' * 30 + ' begin EPOCH ' + str(EPOCH + 1) + '=' * 30)
                train_info = self.train_one_epoch(EPOCH+1)
                val_info = self.val_on_epoch_end(EPOCH+1)

                # lr schedule
                if EPOCH <= self.warmup_epochs:
                    self.optimizer.param_groups[0]['lr'] = self.base_lr * EPOCH / self.warmup_epochs
                else:
                    self.scheduler.step()

                self.write_to_txt(EPOCH, txt_path=cur_txt_path, train_info=train_info, val_info=val_info)

                # 当训练次数超过最低epoch时，其中early_stop策略
                if (EPOCH + 1) > self.opts.min_train_epoch:

                    self.early_stopping(EPOCH + 1, self.ped_model, self.optimizer, val_info, scheduler=None)

                    if self.early_stopping.early_stop:
                        print(f'Early Stopping!')
                        break

    def write_to_txt(self, epoch, txt_path, train_info, val_info):
        train_msg = 'Train: ' + self.get_print_msg(info_dict=train_info)
        val_msg = 'Val: ' + self.get_print_msg(info_dict=val_info)
        with open(txt_path, 'a') as f:
            f.write(f'------------------------------ Epoch: {epoch} ------------------------------\n')
            f.write(train_msg)
            f.write(val_msg)

    @staticmethod
    def _get_msg_format(key):
        '''
            tool_function，对 accuracy 和 loss 进行输出时设置不同的打印位数
        '''

        if 'loss' in key:
            return '{:.6f}'
        if 'accuracy' in key or 'bc' in key:
            return '{:.4f}'
        else:
            return '{}'

    def get_print_msg(self, info_dict):
        msg = ', '.join([f"{k}: {self._get_msg_format(k).format(v)}" for k, v in info_dict.items()]) + '\n'
        return msg



def analyze_info(txt_path):
    '''
    curves:
        loss - epoch
        balanced acc - epoch
    图中标注：收敛时的epoch，最佳balanced acc on val
    '''

    epochs = []
    train_losses = []
    train_ba = []
    val_losses = []
    val_ba = []

    with open(txt_path, 'r') as f:
        data = f.read()

        # 先提取combination信息
        combination_match = re.search(r'^Combination:\s*(.+)$', data, re.MULTILINE)
        comb_name = combination_match.group(1) if combination_match else "Unknown"

        pattern = r'Epoch: (\d+)[\s\S]*?Train: loss: ([\d.]+), balanced_accuracy: ([\d.]+)[\s\S]*?Val: loss: ([\d.]+), balanced_accuracy: ([\d.]+)'
        matches = re.findall(pattern, data)

        epochs = [int(match[0]) for match in matches]
        train_losses = [float(match[1]) for match in matches]
        train_accs = [float(match[2]) for match in matches]
        val_losses = [float(match[3]) for match in matches]
        val_accs = [float(match[4]) for match in matches]

    # 子图1：loss曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2, marker='o')
    plt.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2, marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    hp_list = comb_name.split('_')
    title_name = f'Batch size:{hp_list[0]}, lr:{hp_list[1]}, optm:{hp_list[2]}, lr scheduler:{hp_list[3]}'
    plt.suptitle(title_name, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()








if __name__ == '__main__':
    # from utils.utils import DotDict
    #
    # opts = {
    #     'ped_model_obj': 'torchvision.models.efficientnet_b0',
    #     'ds_name_list': ['D1'],
    #     'rand_seed': 82,
    #     'min_train_epoch': 3,
    #     'model_weights': None,
    #     'isTrain': True
    # }
    # opts = DotDict(opts)
    # aa = HPSelection(opts)
    # aa.hp_search()

    txt_path = r'D:\my_phd\on_git\ML-Classifier-Generalization\HPcomb\hp_txt\1.txt'
    analyze_info(txt_path)





















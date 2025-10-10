import torch, os
import numpy as np


class EarlyStopping():
    def __init__(self, callback_path,
                 top_k=2,
                 cur_epoch=0,
                 monitored_metric='loss',
                 # metric_val=-np.inf,
                 patience=10,
                 delta=0.00001):
        '''
            saving [top_k] best perform models and early stop training when model don't improve for [patience] epochs
            callback_save_path: 保存模型的文件夹
            :param top_k: 保存几个最好模型
            :param patience: 当监控的 metric 连续 patience 个 epoch 不增加，则触发early stopping
            :param delta: 监控metric增加的最小值，当超过该值的时候表示模型有进步
        '''

        self.model_save_dir = callback_path
        self.top_k = top_k

        self.save_prefix = callback_path.split(os.sep)[-1]
        self.cur_epoch = cur_epoch
        self.monitored_metric = monitored_metric
        if monitored_metric == 'loss':
            self.monitored_metric_value = np.inf
        else:
            self.monitored_metric_value = -np.inf

        self.patience = patience
        self.counter = 0            # 记录loss不变的epoch数目
        self.early_stop = False     # 是否停止训练
        self.delta = delta

        print('-' * 20 + ' Early Stopping Info ' + '-' * 20)
        print(f'Create early stopping, monitoring [validation {self.monitored_metric}] changes')
        print(f'The best {self.top_k} models will be saved to {self.model_save_dir}')
        print(f'File saving format: {self.save_prefix}_[epoch]_[{self.monitored_metric}].pth')
        print(f'Early Stop with patience: {self.patience}')

        msg = f'The best {self.top_k} models will be saved to {self.model_save_dir}\n'
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)

    def __call__(self, epoch, model, optimizer, val_epoch_info, scheduler=None):

        improved_flag = True
        self.cur_epoch = epoch
        cur_lr = optimizer.param_groups[0]['lr']
        print(f'Current lr: {cur_lr}')

        if self.monitored_metric in ['accuracy', 'balanced_accuracy']:
            if val_epoch_info[self.monitored_metric] < self.monitored_metric_value + self.delta:       # 表现没有提升的情况
                self.counter += 1
                improved_flag = False
        elif self.monitored_metric == 'loss':
            if val_epoch_info[self.monitored_metric] > self.monitored_metric_value + self.delta:    # 表现没有提升的情况
                self.counter += 1
                improved_flag = False
        else:
            raise ValueError('Wrong monitored metrics!')

        # 表现提升的情况
        if improved_flag:
            metrics = [self.monitored_metric_value, val_epoch_info[self.monitored_metric]]
            self.save_checkpoint(model=model, metrics=metrics, optimizer=optimizer, ckpt_dir=self.model_save_dir, scheduler=scheduler)
            self.counter = 0
        else:
            print(f'Performance Not Improved on Epoch {epoch}. EarlyStopping counter: {self.counter} / {self.patience}')

        # 根据counter判断是否设置停止flag
        if self.counter >= self.patience:
            self.early_stop = True

        # Wring Earlystop Info
        msg = f"Epoch:{epoch}, overall counter:{self.counter}/{self.patience}, current lr: {cur_lr:.8f}\n"
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)


    def del_redundant_weights(self, ckpt_dir):
        all_weights_temp = os.listdir(ckpt_dir)
        all_weights = []
        for weights in all_weights_temp:
            if weights.endswith('.pth'):
                all_weights.append(weights)

        # 按存储格式来： save_name = prefix_{epoch}_{balanced_acc/loss}.pth
        if len(all_weights) > self.top_k - 1:
            sorted = []
            for weight in all_weights:
                val_acc = weight.split('-')[-1]
                sorted.append((weight, val_acc))

            if self.monitored_metric == 'balanced_accuracy':
                sorted.sort(key=lambda w: w[1], reverse=False)
            else:
                sorted.sort(key=lambda w: w[1], reverse=True)

            print('After sorting:', sorted)

            del_path = os.path.join(self.model_save_dir, sorted[0][0])
            os.remove(del_path)
            print('Del file:', del_path)


    def save_checkpoint(self, model, metrics, optimizer, ckpt_dir, scheduler=None):
        print(f'Performance [{self.monitored_metric}] better ({metrics[0]} --> {metrics[1]}). Saving Model.')

        self.del_redundant_weights(ckpt_dir)
        save_name = f"{self.save_prefix}-{self.cur_epoch:02d}-{metrics[1]:.4f}.pth"     # 格式：prefix_{epoch}_{balanced_acc}.pth
        self.monitored_metric_value = metrics[1]

        checkpoint = {
            'epoch': self.cur_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_bc': self.monitored_metric_value,
            'lr': scheduler.get_last_lr() if scheduler is not None else 0,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else '',
        }

        save_path = os.path.join(ckpt_dir, save_name)
        torch.save(checkpoint, save_path)



class Model_Logger():
    '''
        用于记录训练过程中的loss，accuracy变化情况
    '''
    def __init__(self, save_dir, model_name, ds_name_list, train_num_info, val_num_info):
        super().__init__()
        self.save_dir = save_dir
        self.model_name = model_name
        self.ds_name_list = ds_name_list
        self.train_num, self.train_nonPed_num, self.train_ped_num = train_num_info      # 获取数据集的总量，各个类别的量
        self.val_num, self.val_nonPed_num, self.val_ped_num = val_num_info
        self.txt_path = os.path.join(self.save_dir, 'Train_info.txt')

        # 注：训练时取消下列注释
        # __stderr__ = sys.stderr  # 将当前默认的错误输出结果保存为__stderr__
        # sys.stderr = open(os.path.join(self.save_dir, 'errorLog.txt'), 'a')  # 将后续的报错信息写入对应的文件中
        # assert not os.path.exists(self.txt_path), f'The {self.txt_path} already exists, please chcek!'

        # 在文件的开头写入训练的信息
        with open(self.txt_path, 'a') as f:
            msg = f'Model: {model_name}, Training on datasets: {self.ds_name_list}\n'
            f.write(msg)

    @staticmethod
    def _get_msg_format(key):
        '''
            tool_function，对 accuracy 和 loss 进行输出时设置不同的打印位数
        '''

        if 'loss' in key:
            return '{:.8f}'
        if 'accuracy' in key or 'bc' in key:
            return '{:.4f}'
        else:
            return '{}'

    def get_print_msg(self, info_dict):
        msg = ', '.join([f"{k}: {self._get_msg_format(k).format(v)}" for k, v in info_dict.items()]) + '\n'
        return msg

    def __call__(self, epoch, training_info, val_info):
        train_msg = 'Train: ' + self.get_print_msg(info_dict=training_info)
        val_msg = 'Val: ' + self.get_print_msg(info_dict=val_info)
        with open(self.txt_path, 'a') as f:
            f.write(f'------------------------------ Epoch: {epoch} ------------------------------\n')
            f.write(train_msg)
            f.write(val_msg)


























import torch, os, shutil
import numpy as np



class EarlyStopping():
    def __init__(self, callback_path,
                 top_k=1,
                 cur_epoch=0,
                 monitored_metric='loss',
                 patience=10,
                 delta=0.00001):
        '''
            saving [top_k] best perform models and early stop training when model don't improve for [patience] epochs
            callback_save_path: 保存模型的文件夹
            :param top_k: 保存几个最好模型
            :param patience: 当监控的 metric 连续 patience 个 epoch 不增加，则触发early stopping
            :param delta: 监控metric增加的最小值，当超过该值的时候表示模型有进步
            按照epoch数字创建文件夹来保存
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
        self.best_weight_dir = None

        print('-' * 20 + ' Early Stopping Info ' + '-' * 20)
        print(f'Create early stopping, monitoring [validation {self.monitored_metric}] changes')
        print(f'The best {self.top_k} models will be saved to {self.model_save_dir}')
        print(f'File saving format: {self.save_prefix}_[epoch]_[{self.monitored_metric}].pth')
        print(f'Early Stop with patience: {self.patience}')

        msg = f'The best {self.top_k} models will be saved to {self.model_save_dir}\n'
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)

    def __call__(self, epoch, enc, clf, fd, val_epoch_info):

        improved_flag = True
        self.cur_epoch = epoch

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
            self.save_checkpoint(enc, clf, fd, metrics=metrics, ckpt_dir=self.model_save_dir)
            self.counter = 0
        else:
            print(f'Performance Not Improved on Epoch {epoch}. EarlyStopping counter: {self.counter} / {self.patience}')

        # 根据counter判断是否设置停止flag
        if self.counter >= self.patience:
            self.early_stop = True

        # Wring Earlystop Info
        msg = f"Epoch:{epoch}, overall counter:{self.counter}/{self.patience}\n"
        with open(os.path.join(self.model_save_dir, 'cb_EarlyStop.txt'), 'a') as f:
            f.write(msg)


    def del_redundant_weights(self, ckpt_dir):
        '''
            模型按 {epoch}_{loss}来保存的，因此删除的时候需要直接删掉整个文件夹
        '''

        # 先整合保存权重的文件夹
        temp = os.listdir(ckpt_dir)
        weights_dir_list = []
        for f_path in temp:
            if os.path.isdir(os.path.join(ckpt_dir, f_path)):
                weights_dir_list.append(f_path)

        if len(weights_dir_list) > self.top_k - 1:
            sorted = []
            for dir_name in weights_dir_list:
                val_acc = dir_name.split('_')[-1]
                sorted.append((dir_name, val_acc))

            if self.monitored_metric == 'balanced_accuracy':
                sorted.sort(key=lambda w: w[1], reverse=False)
            else:
                sorted.sort(key=lambda w: w[1], reverse=True)
            print('After sorting:', sorted)

            del_path = os.path.join(self.model_save_dir, sorted[0][0])
            shutil.rmtree(del_path)
            print('Del file:', del_path)


    def save_checkpoint(self, enc, clf, fd, metrics, ckpt_dir):
        print(f'Performance [{self.monitored_metric}] better ({metrics[0]} --> {metrics[1]}). Saving Model.')

        self.del_redundant_weights(ckpt_dir)
        # save_name = f"{self.save_prefix}-{self.cur_epoch:02d}-{metrics[1]:.5f}.pth"     # 格式：prefix_{epoch}_{balanced_acc}.pth
        self.monitored_metric_value = metrics[1]        # 更新最优值，用于后续比较

        self.best_weight_dir = os.path.join(ckpt_dir, f'{self.cur_epoch}_{metrics[1]:.5f}')
        if not os.path.exists(self.best_weight_dir):
            os.makedirs(self.best_weight_dir)
        torch.save(enc.state_dict(), os.path.join(self.best_weight_dir, f'dann_feature_{self.cur_epoch:02d}.pt'))
        torch.save(clf.state_dict(), os.path.join(self.best_weight_dir, f'dann_label_{self.cur_epoch:02d}.pt'))
        torch.save(fd.state_dict(), os.path.join(self.best_weight_dir, f'dann_domain_{self.cur_epoch:02d}.pt'))


class Model_Logger():
    '''
        用于记录训练过程中的loss，accuracy变化情况
    '''
    def __init__(self, callback_dir, model_name, ds_name_list):
        super().__init__()
        self.callback_dir = callback_dir
        self.model_name = model_name
        self.ds_name_list = 'to'.join(ds_name_list)
        self.txt_path = os.path.join(self.callback_dir, 'Train_info.txt')

        # 在文件的开头写入训练的信息
        with open(self.txt_path, 'a') as f:
            msg = f'Model: {model_name}, Training on datasets: {self.ds_name_list}\n'
            f.write(msg)

    @staticmethod
    def _get_msg_format(key):
        '''
            tool_function，对 accuracy 和 loss 进行输出时设置不同的打印位数
        '''
        if key in ['loss', 'accuracy', 'balanced_accuracy', 'ba']:
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




























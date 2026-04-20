# https://github.com/NaJaeMin92/pytorch-DANN

import torch, copy, random
import torch.nn as nn
import os
from torch import optim
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.dann_utils import adjust_alpha, DotDict
from implemented_models.dann import Feature_extractor, Label_classifier, Domain_Classifier
from data.dataset import my_dataset
from data.dann_dataset import noise_dataset
from configs.ds_path import DEVICE
from training_func.dann_callbacks import EarlyStopping, Model_Logger


class DANN_Trainer(object):
    def __init__(self, args):
        self.args = args

        # 加载模型
        self.feature_model = Feature_extractor().to(DEVICE)
        self.label_model = Label_classifier().to(DEVICE)
        self.domain_model = Domain_Classifier().to(DEVICE)

        # # 损失函数，训练和测试都需要计算loss
        # self.ce = nn.CrossEntropyLoss().to(DEVICE)

        # 分别为 label_model 和 domain_model 设置loss
        self.label_loss = nn.CrossEntropyLoss().to(DEVICE)
        self.domain_loss = nn.CrossEntropyLoss().to(DEVICE)

        self.best_ba = 0    # balanced acc
        self.time_taken = None

        self.best_weight_dir = None

        if self.args.isTrain:
            self.train_setup()
        else:
            # 若是测试，则创建 test 文件夹用于存储结果
            self.callback_save_path = os.path.join(os.getcwd(), 'Test')
            if not os.path.exists(self.callback_save_path):
                os.mkdir(self.callback_save_path)
            print(f'Test saving dir:{self.callback_save_path}')

        self.print_args()

    def train_setup(self):
        '''
            初始化训练的各种参数
        '''

        # ********** 用seed固定GPU **********
        torch.manual_seed(self.args.cur_seed)
        np.random.seed(self.args.cur_seed)
        random.seed(self.args.cur_seed)
        if DEVICE != 'cpu':
            torch.cuda.manual_seed(self.args.cur_seed)
            # 确保CuDNN的确定性行为
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ********** 加载 source / target data **********
        self.s_train_dataset = my_dataset(ds_name_list=self.args.source, path_key=self.args.path_key, txt_name=self.args.train_txt)
        self.s_train_loader = DataLoader(self.s_train_dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=False)

        self.s_val_dataset = my_dataset(ds_name_list=self.args.source, path_key=self.args.path_key, txt_name=self.args.val_txt)
        self.s_val_loader = DataLoader(self.s_val_dataset, batch_size=self.args.val_batch_size, shuffle=False, drop_last=False)

        # # 用random noise代替target domain，后来实验验证不可行
        # self.t_train_dataset = noise_dataset(num=len(self.s_train_dataset))
        # self.t_train_loader = DataLoader(self.t_train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last)
        #
        # self.t_val_dataset = noise_dataset(num=len(self.s_val_dataset))
        # self.t_val_loader = DataLoader(self.t_val_dataset, batch_size=128, shuffle=False, drop_last=self.drop_last)

        # 用真实数据作为target domain
        self.t_train_dataset = my_dataset(ds_name_list=self.args.target, path_key=self.args.path_key, txt_name=self.args.train_txt)
        self.t_train_loader = DataLoader(self.t_train_dataset, batch_size=self.args.train_batch_size, shuffle=True, drop_last=False)

        self.t_val_dataset = my_dataset(ds_name_list=self.args.target, path_key=self.args.path_key, txt_name=self.args.val_txt)
        self.t_val_loader = DataLoader(self.t_val_dataset, batch_size=self.args.val_batch_size, shuffle=False, drop_last=False)

        # ********** callback **********  命名规则为: DANN{source}{target}
        self.callback_save_dir = f'DANN{self.args.source[0]}{self.args.target[0]}_{str(self.args.cur_seed)}'
        self.callback_save_path = os.path.join(os.getcwd(), self.callback_save_dir)
        print(f'Callback_save_dir:{self.callback_save_path}')
        if not os.path.exists(self.callback_save_path):
            os.mkdir(self.callback_save_path)

        self.early_stopping = EarlyStopping(self.callback_save_path, top_k=self.args.top_k, cur_epoch=0, patience=self.args.patience, monitored_metric=self.args.monitored_metric)
        self.model_logger = Model_Logger(callback_dir=self.callback_save_path, model_name='DANN', ds_name_list=[self.args.source[0], self.args.target[0]])

        # ********** loss & scheduler **********
        self.optimizer = torch.optim.RMSprop(params=list(self.feature_model.parameters()) + list(self.label_model.parameters()) + list(self.domain_model.parameters()), lr=0.01, weight_decay=1e-5, eps=0.001)


    def print_args(self):
        '''
            参数打印并保存到txt文件中
        '''
        print('-' * 40 + ' Args ' + '-' * 40)

        info = []
        for k, v in vars(self.args).items():
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


    def val_on_epoch_end(self, data_loader, val_dataset, epoch):
        self.feature_model.eval()
        self.label_model.eval()
        self.domain_model.eval()

        y_true = []
        y_pred = []
        total_loss_sum = 0.0

        domain_true = []
        domain_pred = []
        domain_correct_num = 0

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(tqdm(data_loader, desc=f'Epoch {epoch} val')):
                images, labels = data_dict['image'].to(DEVICE), data_dict['ped_label'].to(DEVICE)

                features = self.feature_model(images)
                logits = self.label_model(features)
                preds = torch.argmax(logits, dim=1)
                # loss_value = self.ce(logits, labels)
                loss_value = self.label_loss(logits, labels)
                # val_loss += loss_value.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

                # 用于计算domain model的准确率
                domain_out = self.domain_model(features)
                domain_label = torch.zeros(size=(images.shape[0],), dtype=torch.long).to(DEVICE)

                domain_true.extend(domain_label.cpu().numpy())
                domain_pred.extend(torch.argmax(domain_out, dim=1).cpu().numpy())

                domain_correct_num += (torch.argmax(domain_out, dim=1) == domain_label).sum()

                # 便于最终计算每个样本的loss
                batch_loss_sum = loss_value.item() * self.args.val_batch_size
                total_loss_sum += batch_loss_sum

        val_bc = balanced_accuracy_score(y_true, y_pred)
        # domain_model_ba = balanced_accuracy_score(domain_true, domain_pred)
        domain_model_acc = domain_correct_num / len(val_dataset)

        average_val_loss = total_loss_sum / len(val_dataset)

        # cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(2))

        val_epoch_info = {
            'label model balanced_accuracy': val_bc,
            'domain model balanced_accuracy': domain_model_acc,
            'loss': average_val_loss
        }
        return DotDict(val_epoch_info)

    def decomp_cm(self, cm):
        '''
            对混淆矩阵进行分解
        '''
        tn, fp, fn, tp = cm.ravel()
        return f'{tn}, {fp}, {fn}, {tp}'

    def test(self):
        '''
            遍历每个数据集对模型进行测试，并将结果保存到Test文件夹中
        '''

        # load model
        for item in os.listdir(self.args.weight_dir):
            weight_path = os.path.join(self.args.weight_dir, item)
            print(f'weight_path: {weight_path}')
            state_dict = torch.load(weight_path, map_location=DEVICE, weights_only=False)
            if item.split('_')[1] == 'feature':
                self.feature_model.load_state_dict(state_dict)
            elif item.split('_')[1] == 'label':
                self.label_model.load_state_dict(state_dict)
            elif item.split('_')[1] == 'domain':
                self.domain_model.load_state_dict(state_dict)
            else:
                print('Not found model for weights!')

        self.feature_model.eval()
        self.label_model.eval()
        self.domain_model.eval()

        write_to_txt = os.path.join(self.callback_save_path, 'Test.txt')
        with open(write_to_txt, 'a') as f:
            f.write('-' * 80 + '\n')
            f.write(f'Testing model weights dir: {os.listdir(self.args.weight_dir)}.\n')
            f.write('ds_name, test_ba, tnr, tpr, tn, fp, fn, tp\n')

        for ds_name in self.args.test_ds_list:

            # load test data
            test_dataset = my_dataset(ds_name_list=[ds_name], path_key=self.args.path_key, txt_name='test.txt')
            test_loader = DataLoader(test_dataset, batch_size=self.args.test_batch_size, shuffle=False)

            y_true = []
            y_pred = []
            nonPed_acc_num = 0
            ped_acc_num = 0
            test_correct_num = 0
            test_loss = 0.0

            test_nonPed_num, test_ped_num = test_dataset.get_ped_cls_num()

            with torch.no_grad():
                for batch_idx, data_dict in enumerate(tqdm(test_loader, desc='Test')):
                    images, ped_labels = data_dict['image'].to(DEVICE), data_dict['ped_label'].to(DEVICE)

                    logits = self.label_model(self.feature_model(images))
                    preds = torch.argmax(logits, dim=1)
                    # loss_value = self.ce(logits, ped_labels)
                    loss_value = self.label_loss(logits, ped_labels)
                    test_loss += loss_value.item()

                    y_true.extend(ped_labels.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

                    nonPed_idx = (ped_labels == 0)
                    nonPed_acc_num += (ped_labels[nonPed_idx] == preds[nonPed_idx]).sum()
                    ped_idx = (ped_labels == 1)
                    ped_acc_num += ((ped_labels[ped_idx] == preds[ped_idx]) * 1).sum()

                test_ba = balanced_accuracy_score(y_true, y_pred)
                test_cm = confusion_matrix(y_true, y_pred)

                test_nonPed_acc = nonPed_acc_num / test_nonPed_num
                test_ped_acc = ped_acc_num / test_ped_num

                msg = f'DS_name:{ds_name}, Balanced accuracy:{test_ba:.4f}\nNon-ped accuracy:{test_nonPed_acc:.4f}({nonPed_acc_num}/{test_nonPed_num}), Ped accuracy:{test_ped_acc:.4f}({ped_acc_num}/{test_ped_num})'
                print(msg)
                print(f'CM on test set:\n{test_cm}')

                print('-' * 40 + 'Test Info' + '-' * 40)

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                print(tn, fp, fn, tp)

                with open(write_to_txt, 'a') as f:
                    f.write(
                        f'{ds_name}, {test_ba:.6f}, {test_nonPed_acc:.4f}, {test_ped_acc:.4f}, {tn}, {fp}, {fn}, {tp}\n')



            # print(f'cm: {test_cm}')
            #
            # with open(self.args.res_save_txt, 'a') as f:
            #     msg = f'model_weights: {self.args.weight_dir}\nds_name: {self.args.test_ds_list[0]}\nTest loss: {test_loss:.4f}\nTest balanced acc: {test_ba:.4f}\ntn, fp, fn, tp: {self.decomp_cm(test_cm)}\n'
            #     print(msg)
            #     f.write(msg)

    # def update_learning_rate(self, epoch):
    #     old_lr = self.optimizer.param_groups[0]['lr']
    #
    #     if epoch <= self.args.warmup_epochs:
    #         self.optimizer.param_groups[0]['lr'] = self.args.base_lr * epoch / self.args.warmup_epochs
    #     else:
    #         self.scheduler.step()
    #
    #     lr = self.optimizer.param_groups[0]['lr']
    #     print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def update_learning_rate(self, epoch):
        # old_lr = self.optimizer.param_groups[0]['lr']

        # warm-up阶段
        if epoch <= self.args.warmup_epochs:  # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.args.base_lr * epoch / self.args.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.args.base_lr * 0.963 ** (epoch / 3)  # gamma=0.963, lr decay epochs=3

        # lr = self.optimizer.param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def train_one_epoch(self, epoch, min_len):
        self.label_model.train()
        self.feature_model.train()
        self.domain_model.train()

        # loss_val = 0.0
        total_loss_sum = 0.0
        y_true = []
        y_pred = []

        domain_true = []
        domain_pred = []

        for batch_idx, (source_dict, target_dict) in tqdm(enumerate(zip(self.s_train_loader, self.t_train_loader)), total=len(self.s_train_loader), desc=f'Epoch {epoch} train'):
            # 调节domain classifier的alpha
            alpha = adjust_alpha(batch_idx, epoch, min_len, self.args.max_train_epochs)

            # 加载数据
            source, s_labels = source_dict['image'].to(DEVICE), source_dict['ped_label'].to(DEVICE)
            target = target_dict['image'].to(DEVICE)

            # label classifier
            s_feature = self.feature_model(source)
            s_out = self.label_model(s_feature)
            s_preds = torch.argmax(s_out, dim=1)

            t_feature = self.feature_model(target)

            # domain classifier
            s_domain_out = self.domain_model(s_feature, alpha=alpha)
            t_domain_out = self.domain_model(t_feature, alpha=alpha)

            # source and target label
            source_domain_label = torch.zeros(size=(source.shape[0],), dtype=torch.long).to(DEVICE)
            target_domain_label = torch.ones(size=(target.shape[0],), dtype=torch.long).to(DEVICE)
            s_domain_err = self.domain_loss(s_domain_out, source_domain_label)
            t_domain_err = self.domain_loss(t_domain_out, target_domain_label)

            # 计算domain model的准确率
            domain_true.extend(source_domain_label.cpu().numpy())
            domain_pred.extend(torch.argmax(s_domain_out, dim=1).cpu().numpy())
            domain_true.extend(target_domain_label.cpu().numpy())
            domain_pred.extend(torch.argmax(t_domain_out, dim=1).cpu().numpy())

            domain_loss = s_domain_err + t_domain_err
            s_label_loss = self.label_loss(s_out, s_labels)
            loss_value = s_label_loss + domain_loss

            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            # 记录在source domain上的accuracy
            y_true.extend(s_labels.cpu().numpy())
            y_pred.extend(s_preds.cpu().numpy())

            # 计算出该batch的loss
            batch_loss_sum = loss_value.item() * self.args.train_batch_size
            total_loss_sum += batch_loss_sum

        train_bc = balanced_accuracy_score(y_true, y_pred)
        average_train_loss = total_loss_sum / len(self.s_train_dataset)

        # 计算domain model的准确率
        domain_model_bc = balanced_accuracy_score(domain_true, domain_pred)

        train_epoch_info = {
            'label model balanced accuracy': train_bc,
            'domain model balanced accuracy': domain_model_bc,
            'loss': average_train_loss
        }

        return train_epoch_info


    def train(self):
        s_iter_per_epoch = len(self.s_train_loader)
        t_iter_per_epoch = len(self.t_train_loader)
        min_len = min(s_iter_per_epoch, t_iter_per_epoch)

        print("Source iters per epoch: %d" % (s_iter_per_epoch))
        print("Target iters per epoch: %d" % (t_iter_per_epoch))
        print("iters per epoch: %d" % (min(s_iter_per_epoch, t_iter_per_epoch)))

        for EPOCH in range(self.args.max_train_epochs):
            print('=' * 30 + ' Begin EPOCH ' + str(EPOCH + 1) + '=' * 30)

            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            train_info = self.train_one_epoch(EPOCH+1, min_len=min_len)
            val_info = self.val_on_epoch_end(self.s_val_loader, self.s_val_dataset, epoch=EPOCH+1)        # 用真实数据作为target

            print(f'Train info:')
            for k, v in train_info.items():
                print(f'{k} - {v:.6f}')
            print(f'Val info:')
            for k, v in val_info.items():
                print(f'{k} - {v:.6f}')

            # print(f'Train loss {train_info["loss"]:.6f}, train_bc:{train_info["balanced_accuracy"]:.4f}')
            # print(f'Val loss {val_info["loss"]:.6f}, val_bc:{val_info["balanced_accuracy"]:.4f}')

            # ------------------------ 调用callbacks ------------------------
            self.model_logger(epoch=EPOCH+1, training_info=train_info, val_info=val_info)
            #
            # ------------------------ 学习率调整 ------------------------
            self.update_learning_rate(EPOCH + 1)

            # DANN为固定epoch，不需要early stop，这里用到early stop callback中的模型保存功能
            if (EPOCH + 1) > self.args.min_train_epochs:
                self.early_stopping(EPOCH + 1, enc=self.feature_model, clf=self.label_model, fd=self.domain_model, val_epoch_info=val_info, model_save=True)

            # 最后一个epoch保存latest model
            if (EPOCH+1) == self.args.max_train_epochs:
                torch.save(self.feature_model.state_dict(), os.path.join(self.callback_save_path, f'dann_feature_lastEpoch.pt'))
                torch.save(self.label_model.state_dict(), os.path.join(self.callback_save_path, f'dann_label_lastEpoch.pt'))
                torch.save(self.domain_model.state_dict(), os.path.join(self.callback_save_path, f'dann_domain_lastEpoch.pt'))



            #     if self.early_stopping.early_stop:
            #         print(f'Early Stopping!')
            #         break


            # # 在低于min train epoch时，每次重置early stop的参数
            # if (EPOCH + 1) <= self.min_epochs:
            #     self.early_stopping.counter = 0
            #     self.early_stopping.early_stop = False
            # else:  # 当训练次数超过最低epoch时，其中early_stop策略
            #     self.early_stopping(EPOCH + 1, enc=self.feature_model, clf=self.label_model, fd=self.domain_model, val_epoch_info=val_info)
            #     if self.early_stopping.early_stop:
            #         print(f'Early Stopping!')
            #         break







































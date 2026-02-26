import torch, copy
import torch.nn as nn
import os
from torch import optim
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils.dann_utils import adjust_alpha, DotDict
from implement_models.dann import Feature_extractor, Label_classifier, Domain_Classifier
from data.dataset import my_dataset
from data.dann_dataset import noise_dataset
from configs.ds_path import DEVICE
from training.callbacks import EarlyStopping, Model_Logger


class DANN_Trainer(object):
    def __init__(self, args):
        self.args = args
        self.print_args()

        self.drop_last = False
        self.batch_size = self.args.batch_size
        self.min_epochs = self.args.min_epochs
        self.max_epochs = self.args.max_epochs
        self.warmup_epochs = self.args.warmup_epochs

        torch.manual_seed(self.args.seed)

        # 加载模型
        self.feature_model = Feature_extractor().to(DEVICE)
        self.label_model = Label_classifier().to(DEVICE)
        self.domain_model = Domain_Classifier().to(DEVICE)

        # self.enc = Feature_extractor().to(DEVICE)
        # self.clf = Label_classifier().to(DEVICE)
        # self.fd = Domain_Classifier().to(DEVICE)

        # 损失函数，训练和测试都需要计算loss
        self.ce = nn.CrossEntropyLoss().to(DEVICE)
        # self.bce = nn.BCELoss()

        self.best_ba = 0    # balanced acc
        self.time_taken = None

        # # 用于 domain classifier训练的label，cbe作为损失函数时
        # self.fake_label = torch.FloatTensor(self.args.batch_size, 1).fill_(0).to(DEVICE)
        # self.real_label = torch.FloatTensor(self.args.batch_size, 1).fill_(1).to(DEVICE)

        self.best_weight_dir = None

        if self.args.isTrain:
            self.train_setup()

    def train_setup(self):
        # 加载data
        self.s_train_dataset = my_dataset(ds_name_list=self.args.source, path_key=self.args.path_key, txt_name=self.args.train_txt)
        self.s_train_loader = DataLoader(self.s_train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last)

        self.s_val_dataset = my_dataset(ds_name_list=self.args.source, path_key=self.args.path_key, txt_name=self.args.val_txt)
        self.s_val_loader = DataLoader(self.s_val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=self.drop_last)

        # # 用random noise代替target domain
        # self.t_train_dataset = noise_dataset(num=len(self.s_train_dataset))
        # self.t_train_loader = DataLoader(self.t_train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last)
        #
        # self.t_val_dataset = noise_dataset(num=len(self.s_val_dataset))
        # self.t_val_loader = DataLoader(self.t_val_dataset, batch_size=128, shuffle=False, drop_last=self.drop_last)

        # 用真实数据作为target domain
        self.t_train_dataset = my_dataset(ds_name_list=self.args.target, path_key=self.args.path_key, txt_name=self.args.train_txt)
        self.t_train_loader = DataLoader(self.t_train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=self.drop_last)

        self.t_val_dataset = my_dataset(ds_name_list=self.args.target, path_key=self.args.path_key, txt_name=self.args.val_txt)
        self.t_val_loader = DataLoader(self.t_val_dataset, batch_size=128, shuffle=False, drop_last=self.drop_last)

        # optimizer
        self.optimizer = torch.optim.RMSprop(params=list(self.feature_model.parameters()) + list(self.label_model.parameters()) + list(self.domain_model.parameters()), lr=0.0, weight_decay=1e-5, eps=0.001)

        # callbacks
        self.early_stopping = EarlyStopping(self.callback_dir, top_k=self.args.top_k, cur_epoch=0, patience=self.args.patience, monitored_metric=self.args.monitored_metric)
        # self.model_logger = Model_Logger(save_dir=self.callback_dir, model_name='DANN', ds_name_list=[self.args.source[0], self.args.target[0]])


    def print_args(self):
        '''
            Printing args to the console & Saing args to .txt file
        '''
        print('-' * 40 + ' Args ' + '-' * 40)
        self.callback_name = 'DANN' + '_' + self.args.source[0] + self.args.target[0] + f'_{self.args.seed}'

        self.callback_dir = os.path.join(os.getcwd(), self.callback_name)
        if not os.path.exists(self.callback_dir):
            os.mkdir(self.callback_dir)
        print(f'Callback dir: {self.callback_dir}')

        with open(os.path.join(self.callback_dir, 'Args.txt'), 'a') as f:
            for k, v in vars(self.args).items():
                msg = f'{k}: {v}'
                print(msg)
                f.write(msg + '\n')

    def val_on_epoch_end(self, data_loader, epoch):
        self.feature_model.eval()
        self.label_model.eval()

        y_true = []
        y_pred = []
        val_loss = 0.0

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(tqdm(data_loader, desc=f'Epoch {epoch} val')):
                images, labels = data_dict['image'].to(DEVICE), data_dict['ped_label'].to(DEVICE)

                logits = self.label_model(self.feature_model(images))
                preds = torch.argmax(logits, dim=1)
                loss_value = self.ce(logits, labels)
                val_loss += loss_value.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_bc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=range(2))

        # print(f'CM on validation set:\n{cm}')

        val_epoch_info = {
            'balanced_accuracy': val_bc,
            'loss': val_loss
        }
        return DotDict(val_epoch_info)

    def decomp_cm(self, cm):
        '''
            对混淆矩阵进行分解
        '''
        tn, fp, fn, tp = cm.ravel()
        return f'{tn}, {fp}, {fn}, {tp}'

    def test(self):
        # load test data
        test_dataset = my_dataset(ds_name_list=self.args.test_ds_list, path_key='Stage6_org', txt_name='test.txt')
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # load model
        for item in os.listdir(self.args.weight_dir):
            weight_path = os.path.join(self.args.weight_dir, item)
            print(f'weight_path: {weight_path}')
            state_dict = torch.load(weight_path, map_location=DEVICE)
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

        # 开始测试
        y_true = []
        y_pred = []
        test_loss = 0.0

        with torch.no_grad():
            for batch_idx, data_dict in enumerate(tqdm(test_loader, desc='Test')):
                images, labels = data_dict['image'].to(DEVICE), data_dict['ped_label'].to(DEVICE)

                logits = self.label_model(self.feature_model(images))
                preds = torch.argmax(logits, dim=1)
                loss_value = self.ce(logits, labels)
                test_loss += loss_value.item()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

            test_ba = balanced_accuracy_score(y_true, y_pred)
            test_cm = confusion_matrix(y_true, y_pred)
            print(f'cm: {test_cm}')

            with open(self.args.test_txt, 'a') as f:
                msg = f'model_weights: {self.args.weight_dir}\nds_name: {self.args.test_ds_list[0]}\nTest loss: {test_loss:.4f}\nTest balanced acc: {test_ba:.4f}\ntn, fp, fn, tp: {self.decomp_cm(test_cm)}\n'
                print(msg)
                f.write(msg)

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
        if epoch <= self.warmup_epochs:  # warm-up阶段
            self.optimizer.param_groups[0]['lr'] = self.args.base_lr * epoch / self.warmup_epochs
        else:
            self.optimizer.param_groups[0]['lr'] = self.args.base_lr * 0.963 ** (epoch / 3)  # gamma=0.963, lr decay epochs=3

        # lr = self.optimizer.param_groups[0]['lr']
        # print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def train_one_epoch(self, epoch, min_len):
        self.label_model.train()
        self.feature_model.train()
        self.domain_model.train()

        loss_val = 0.0
        y_true = []
        y_pred = []

        for batch_idx, (source_dict, target_dict) in tqdm(enumerate(zip(self.s_train_loader, self.t_train_loader)),
                                                          total=len(self.s_train_loader),
                                                          desc=f'Epoch {epoch} train'):
            # 调节domain classifier的alpha
            # total_iters += 1
            alpha = adjust_alpha(batch_idx, epoch, min_len, self.max_epochs)

            # 加载数据
            source, s_labels = source_dict['image'].to(DEVICE), source_dict['ped_label'].to(DEVICE)
            target = target_dict['image'].to(DEVICE)

            # label classifier
            s_feature = self.feature_model(source)
            s_out = self.label_model(s_feature)
            s_preds = torch.argmax(s_out, dim=1)

            t_feature = self.feature_model(target)
            t_out = self.label_model(t_feature)

            # domain classifier
            s_domain_out = self.domain_model(s_feature, alpha=alpha)
            t_domain_out = self.domain_model(t_feature, alpha=alpha)

            # 两个classifier都用交叉熵损失
            real_label = torch.ones(size=(source.shape[0],), dtype=torch.long).to(DEVICE)
            fake_label = torch.zeros(size=(target.shape[0],), dtype=torch.long).to(DEVICE)
            s_domain_err = self.ce(s_domain_out, real_label)
            t_domain_err = self.ce(t_domain_out, fake_label)

            domain_loss = s_domain_err + t_domain_err
            s_label_loss = self.ce(s_out, s_labels)
            loss = s_label_loss + domain_loss
            loss_val += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录在source domain上的accuracy
            y_true.extend(s_labels.cpu().numpy())
            y_pred.extend(s_preds.cpu().numpy())

        train_bc = balanced_accuracy_score(y_true, y_pred)

        train_epoch_info = {
            'balanced_accuracy': train_bc,
            'loss': loss_val
        }

        return train_epoch_info


    def train(self):
        s_iter_per_epoch = len(self.s_train_loader)
        t_iter_per_epoch = len(self.t_train_loader)
        min_len = min(s_iter_per_epoch, t_iter_per_epoch)
        # total_iters = 0

        print("Source iters per epoch: %d" % (s_iter_per_epoch))
        print("Target iters per epoch: %d" % (t_iter_per_epoch))
        print("iters per epoch: %d" % (min(s_iter_per_epoch, t_iter_per_epoch)))

        # self.optimizer = optim.Adam(params=list(self.feature_model.parameters()) + list(self.label_model.parameters()) + list(self.domain_model.parameters()), lr=self.base_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_epochs - self.warmup_epochs)


        for EPOCH in range(self.max_epochs):

            # 在epoch开始之前调节lr
            self.update_learning_rate(EPOCH + 1)

            train_info = self.train_one_epoch(EPOCH+1, min_len=min_len)
            # val_info = self.val_on_epoch_end(self.t_val_loader, epoch=EPOCH+1)        # 用真实数据作为target
            val_info = self.val_on_epoch_end(self.s_val_loader, epoch=EPOCH+1)          # 用noise作为target

            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}')
            print(f'Train loss {train_info["loss"]:.6f}, train_bc:{train_info["balanced_accuracy"]:.4f}')
            print(f'Val loss {val_info["loss"]:.6f}, val_bc:{val_info["balanced_accuracy"]:.4f}')

            # callback, model logger
            self.model_logger(epoch=EPOCH+1, training_info=train_info, val_info=val_info)

            # 在低于min train epoch时，每次重置early stop的参数
            if (EPOCH + 1) <= self.min_epochs:
                self.early_stopping.counter = 0
                self.early_stopping.early_stop = False
            else:  # 当训练次数超过最低epoch时，其中early_stop策略
                self.early_stopping(EPOCH + 1, enc=self.feature_model, clf=self.label_model, fd=self.domain_model, val_epoch_info=val_info)
                if self.early_stopping.early_stop:
                    print(f'Early Stopping!')
                    break







































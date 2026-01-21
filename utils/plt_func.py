'''
    包含一系列用于绘制的函数

    其中绘制训练曲线函数的相关信息：
        根据训练过程中的 Train_info.txt文件格式：
            Model: xxx, Training on datasets: xxx
            ------------------------------ Epoch: 1 ------------------------------
            Train: accuracy: xxx, balanced_accuracy: xxx, loss: xxx
            Val: accuracy: xxx, balanced_accuracy: xxx, loss: xxx
            ------------------------------ Epoch: 2 ------------------------------
            ...
        绘制train/val的 balanced acc和loss随epoch的变化曲线
'''
import os.path
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


def extract_training_data(txt_path):
    # 读取文件内容
    with open(txt_path, 'r') as f:
        content = f.read()

    # 正则表达式模式
    # 匹配epoch数字
    epoch_pattern = r'Epoch:\s*(\d+)'
    # 匹配train的balanced accuracy和loss
    train_pattern = r'Train:.*?balanced_accuracy:\s*([\d.]+).*?loss:\s*([\d.]+)'
    # 匹配val的balanced accuracy和loss
    val_pattern = r'Val:.*?balanced_accuracy:\s*([\d.]+).*?loss:\s*([\d.]+)'

    # 提取数据
    epochs = re.findall(epoch_pattern, content)
    train_matches = re.findall(train_pattern, content, re.DOTALL)
    val_matches = re.findall(val_pattern, content, re.DOTALL)

    # 转换为数值类型
    epochs = [int(epoch) for epoch in epochs]
    train_balanced_acc = [float(match[0]) for match in train_matches]
    train_loss = [float(match[1]) for match in train_matches]
    val_balanced_acc = [float(match[0]) for match in val_matches]
    val_loss = [float(match[1]) for match in val_matches]

    return {
        'epochs': epochs,
        'train_balanced_acc': train_balanced_acc,
        'train_loss': train_loss,
        'val_balanced_acc': val_balanced_acc,
        'val_loss': val_loss
    }


def plot_training_curves(data, save_path=None):
    """
        绘制训练曲线图
        data: 包含训练数据的字典
        save_path: 图片保存路径，如果为None则显示图片
    """
    epochs = data['epochs']

    # 创建图形和第一个y轴
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制balanced accuracy曲线（左y轴）
    color_acc = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Balanced Accuracy', color=color_acc, fontsize=12)

    # 绘制train balanced accuracy
    ax1.plot(epochs, data['train_balanced_acc'],
             color='blue', marker='o', linestyle='-',
             linewidth=2, markersize=6, label='Train Balanced Accuracy')

    # 绘制val balanced accuracy
    ax1.plot(epochs, data['val_balanced_acc'],
             color='cyan', marker='s', linestyle='--',
             linewidth=2, markersize=6, label='Val Balanced Accuracy')

    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])  # accuracy通常在0-1之间

    # 创建第二个y轴用于loss
    ax2 = ax1.twinx()
    color_loss = 'tab:red'
    ax2.set_ylabel('Loss', color=color_loss, fontsize=12)

    # 绘制train loss
    ax2.plot(epochs, data['train_loss'],
             color='red', marker='^', linestyle='-',
             linewidth=2, markersize=6, label='Train Loss')

    # 绘制val loss
    ax2.plot(epochs, data['val_loss'],
             color='orange', marker='d', linestyle='--',
             linewidth=2, markersize=6, label='Val Loss')

    ax2.tick_params(axis='y', labelcolor=color_loss)

    # 添加标题和图例
    plt.title('Training Progress: Balanced Accuracy and Loss', fontsize=14, fontweight='bold')

    # 合并图例（需要将两条轴的图例合并）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10, framealpha=0.9)

    # 调整布局
    fig.tight_layout()

    # 保存或显示图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_cm(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, cm_save_dir=None):
    """
        y_true: 真实标签
        y_pred: 预测标签
        classes: 类别名称列表
        normalize: 是否归一化显示百分比
        title: 图表标题
        cmap: 颜色映射
    """

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)

    # 绘制热力图
    ax = sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                     cmap=cmap, square=True,
                     xticklabels=classes, yticklabels=classes,
                     cbar_kws={"shrink": 0.8})

    # 设置标题和标签
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    else:
        title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
        ax.set_title(title, fontsize=16, pad=20)

    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)

    # 保存或显示图片
    if cm_save_dir:
        save_path = os.path.join(cm_save_dir, 'cm.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")

    # 调整布局
    plt.tight_layout()
    plt.show()






if __name__ == '__main__':
    print('start')

    # txt_path = r'D:\my_phd\Model_Weights\Stage6\new_dataset\trainOnMultiDataset\efficientNetB0_D1_D3_33_Baseline\Train_info.txt'
    # data = extract_training_data(txt_path)
    # plot_training_curves(data)

    # 绘制混淆矩阵

    y_true = ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']
    y_pred = ['A', 'B', 'B', 'A', 'C', 'C', 'A', 'B', 'C']
    classes = ['A', 'B', 'C']
    # plot_cm(y_true, y_pred, classes, normalize=False, title='Confusion Matrix')
    # plot_cm(y_true, y_pred, classes, normalize=True, title='Confusion Matrix')

    # from torchvision import models
    # from torch.utils.data import DataLoader
    #
    # from utils import load_model
    # from data.dataset import my_dataset
    # from tqdm import tqdm
    #
    # # model
    # ds_model = models.efficientnet_b0(weights=None, num_classes=3)
    # ds_weights_path = r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth'
    # ds_model = load_model(ds_model, weights_path=ds_weights_path)
    #
    # ds_model.eval()
    #
    # # data
    # test_dataset = my_dataset(ds_name_list=['D1', 'D2', 'D3'], path_key='Stage6_org', txt_name='test.txt')
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    #
    # y_true = []
    # y_pred = []
    #
    # with torch.no_grad():
    #     for data_dict in tqdm(test_loader):
    #         print(data_dict.keys())
    #         images = data_dict['image']
    #         labels = data_dict['ds_label']
    #
    #         print(labels)
    #
    #         break









































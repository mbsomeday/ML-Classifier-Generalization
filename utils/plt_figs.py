'''
    根据训练过程中的 Train_info.txt文件格式：
        Model: xxx, Training on datasets: xxx
        ------------------------------ Epoch: 1 ------------------------------
        Train: accuracy: xxx, balanced_accuracy: xxx, loss: xxx
        Val: accuracy: xxx, balanced_accuracy: xxx, loss: xxx
        ------------------------------ Epoch: 2 ------------------------------
        ...
    绘制train/val的 balanced acc和loss随epoch的变化曲线
'''
import re
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    txt_path = r'D:\my_phd\Model_Weights\Stage6\new_dataset\trainOnMultiDataset\efficientNetB0_D1_D3_33_Baseline\Train_info.txt'

    data = extract_training_data(txt_path)
    plot_training_curves(data)






































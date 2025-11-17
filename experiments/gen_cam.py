# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import random, os, argparse
from tqdm import tqdm
from torchcam.methods.gradient import GradCAM, GradCAMpp, LayerCAM
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from torchcam.utils import overlay_mask
import numpy as np

from utils.utils import load_model, save_image_tensor
from data.dataset import my_dataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--model_weights', type=str)

    args = parser.parse_args()

    return args

args = get_args()


torch.manual_seed(13)
batch_size = 1
save_dir = args.save_dir
ds_name_list = args.ds_name_list
model_weights = args.model_weights

# save_dir = r'D:\my_phd\on_git\ML-Classifier-Generalization\aa_test'
# ds_name_list = ['D1']

# 加载模型和数据等
ped_model = efficientnet_b0(num_classes=2, weights=None)
# model_weights = r'/kaggle/input/stage6-weights-baseline/efficientNetB0_D1_3_Baseline-18-4.63655.pth'
# model_weights = r'D:\my_phd\Model_Weights\Stage6\new_dataset\baselines\D1\efficientNetB0_D1_3_Baseline-18-4.63655.pth'
ped_model = load_model(ped_model, model_weights)
ped_model.eval()

train_dataset = my_dataset(ds_name_list=ds_name_list, path_key='Stage6_org', txt_name='train.txt')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

cam_extractor = GradCAM(ped_model, target_layer='features')
# layerCam_extractor = LayerCAM(ped_model, target_layer='features')   # todo: 将不同层的特征图结合
cur_extractor = cam_extractor

# transformers
plt_transformer = transforms.ToPILImage()
tensor_transformer = transforms.ToTensor()
plt_resize = transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)

gaussion_trans = transforms.GaussianBlur(kernel_size=9, sigma=(1, 3))
gray_trans = transforms.Grayscale()
color_trans = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02)
sharp_trans = transforms.RandomAdjustSharpness(sharpness_factor=5)
qeualize_trans = transforms.RandomEqualize()

trans_list = [gaussion_trans, gray_trans, color_trans, sharp_trans, qeualize_trans]
trans_name_list = ['gaussian', 'gray', 'color', 'sharp', 'qeualize']
cur_trans = gaussion_trans

for data_dict in tqdm(train_loader):
    # print(data_dict.keys())

    image = data_dict['image']
    image_path = data_dict['img_path'][0]
    img_name = data_dict['img_name'][0]
    ped_label = data_dict['ped_label']

    cls_name = image_path.split(os.sep)[-2]

    ped_out = ped_model(image)
    cam = cur_extractor(ped_out.squeeze(0).argmax().item(), ped_out)
    plt_cam = plt_transformer(cam[0])
    resized_cam = plt_resize(plt_cam)

    cam_array = np.array(resized_cam)
    cam_mask = cam_array >= (0.2 * cam_array.max())       # 将图像感兴趣的区域保留，
    # cam_mask = cam_array > 0
    plt_mask = cam_mask * 1.0
    plt_mask = plt_mask[np.newaxis, :]

    # 保存转换过的图片
    trans_idx = random.randint(0, len(trans_list))
    # 保存的名字
    new_image_name = os.path.splitext(img_name)[0] + '_' + trans_name_list[trans_idx] + os.path.splitext(img_name)[-1]
    save_path = os.path.join(save_dir, cls_name, new_image_name)
    # 进行的图片变换
    cur_trans = trans_list[trans_idx]
    aug_image = cur_trans(plt_transformer(image[0]))
    comb_image = transforms.ToTensor()(aug_image) * (1 - plt_mask) + image[0] * plt_mask
    save_image_tensor(comb_image.unsqueeze(0), save_path)

    # 保存原始图片
    save_path = os.path.join(save_dir, cls_name, img_name)
    save_image_tensor(image[0], save_path)


    # 循环所有的trans
    # for trans_idx in range(len(trans_list)):
    #     # 保存的名字
    #     new_image_name = os.path.splitext(img_name)[0] + '_' + trans_name_list[trans_idx] + os.path.splitext(img_name)[-1]
    #     save_path = os.path.join(save_dir, cls_name, new_image_name)
    #     # 进行的图片变换
    #     cur_trans = trans_list[trans_idx]
    #     aug_image = cur_trans(plt_transformer(image[0]))
    #     comb_image = transforms.ToTensor()(aug_image) * (1 - plt_mask) + image[0] * plt_mask
    #     save_image_tensor(comb_image.unsqueeze(0), save_path)


    # aug_image = cur_trans(plt_transformer(image[0]))
    # comb_image = transforms.ToTensor()(aug_image) * (1 - plt_mask) + image[0] * plt_mask
    # # print(type(comb_image))
    # save_image_tensor(comb_image.unsqueeze(0), save_path)


    # plt_imgs = 3
    #
    # # 创建一个包含两个子图的网格
    # plt.subplot(1, plt_imgs, 1)
    # plt.imshow(plt_transformer(image[0]))
    # plt.title('org')
    # plt.axis('off')  # 关闭坐标轴
    #
    # plt.subplot(1, plt_imgs, 2)
    # plt.imshow(plt_mask[0], cmap='gray')
    # plt.title('mask')
    #
    # plt.subplot(1, plt_imgs, 3)
    # plt.imshow(plt_transformer(comb_image))
    # # plt.title('grad-cam')
    # plt.axis('off')  # 关闭坐标轴
    #
    # # 显示图片
    # plt.show()


    # break


# def gen_comb_txt():
#     ped_dir = r'D:\my_phd\on_git\ML-Classifier-Generalization\aa_test\pedestrian'
#     nonPed_dir = r'D:\my_phd\on_git\ML-Classifier-Generalization\aa_test\nonPedestrian'
#     aug_txt_path = r'D:\my_phd\on_git\ML-Classifier-Generalization\aa_test\comb_augmentation_train.txt'
#
#     ped_list = os.listdir(ped_dir)
#     nonPed_list = os.listdir(nonPed_dir)
#     with open(aug_txt_path, 'a') as f:
#         for item in ped_list:
#             item = os.path.join('augmentation_train', 'pedestrian', item) + ' 1\n'
#             f.write(item)
#         for item in nonPed_list:
#             item = os.path.join('augmentation_train', 'nonPedestrian', item) + ' 0\n'
#             f.write(item)

# gen_comb_txt()














































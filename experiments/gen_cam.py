# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import random, os, argparse
from PIL import Image
from tqdm import tqdm
from torch import nn, optim
from torchcam.methods.gradient import GradCAM, GradCAMpp, LayerCAM
from torchcam.methods.activation import ScoreCAM
from torchvision.models import efficientnet_b0
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode
from torchcam.utils import overlay_mask
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

from utils.utils import load_model, save_image_tensor
from data.dataset import my_dataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default=r'D:\my_phd\on_git\ML-Classifier-Generalization\aa_test')
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\baselines\D1\efficientNetB0_D1_3_Baseline-18-4.63655.pth')
    parser.add_argument('--perturb_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_ecp\Perturbations\Attack_FastGradient_test')
    parser.add_argument('--txt_name', type=str, default='train.txt')

    args = parser.parse_args()

    return args

args = get_args()


torch.manual_seed(43)
batch_size = 1
txt_name = args.txt_name
save_dir = args.save_dir
ds_name_list = args.ds_name_list
model_weights = args.model_weights
perturb_dir = args.perturb_dir

# 加载模型和数据等
ped_model = efficientnet_b0(num_classes=2, weights=None)
ped_model = load_model(ped_model, model_weights)
ped_model.eval()

# for name, _ in ped_model.named_modules():
#     print(f'name: {name}')

train_dataset = my_dataset(ds_name_list=ds_name_list, path_key='Stage6_org', txt_name=txt_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

grad_layer = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8']
cam_extractor = GradCAM(ped_model, target_layer='features')
layerCam_extractor = LayerCAM(ped_model, target_layer=grad_layer)
cur_extractor = layerCam_extractor

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

# # attack
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(ped_model.parameters(), lr=0.01)
# classifier = PyTorchClassifier(
#     model=ped_model,
#     loss=criterion,
#     optimizer=optimizer,
#     input_shape=(3, 224, 224),
#     nb_classes=2,
# )
# attack = FastGradientMethod(estimator=classifier, eps=0.1)

for data_dict in tqdm(train_loader):
    # print(data_dict.keys())

    image = data_dict['image']
    image_path = data_dict['img_path'][0]
    img_name = data_dict['img_name'][0]
    ped_label = data_dict['ped_label']
    # print(f'img_name:{img_name}， ped_label:{ped_label}')

    cls_name = image_path.split(os.sep)[-2]

    # print(f'cls_name:{cls_name}')
    ped_out = ped_model(image)
    cam = cur_extractor(ped_out.squeeze(0).argmax().item(), ped_out)

    # 将layercam不同特征图合并
    resized_hp = []
    for hp in cam:
        hp = plt_transformer(hp)
        hp = hp.resize((224, 224), resample=Image.BICUBIC)
        cur_hp = tensor_transformer(hp).unsqueeze(0)
        resized_hp.append(cur_hp)
    vis_heatmaps = torch.cat(resized_hp, dim=0)
    comb_layercam = torch.sum(vis_heatmaps, 0).unsqueeze(0)
    (cam_min, cam_max) = (comb_layercam.min(), comb_layercam.max())
    norm_cam = (comb_layercam - cam_min) / (((cam_max - cam_min) + 1e-08)).data

    # print(f'vis_heatmaps: {type(vis_heatmaps)}, {vis_heatmaps.shape}')
    # print(f'comb_layercam: {comb_layercam.shape}')

    plt_cam = plt_transformer(norm_cam[0])

    # plt_cam = plt_transformer(cam[0])
    resized_cam = plt_resize(plt_cam)

    cam_array = np.array(resized_cam)
    # cam_mask = cam_array < (0.2 * cam_array.max())  # 将图像不感兴趣的区域保留
    cam_mask = cam_array >= (0.4 * cam_array.max())       # 将图像感兴趣的区域保留
    plt_mask = cam_mask * 1.0
    plt_mask = plt_mask[np.newaxis, :]

    # 加载perturb图片
    perturb_image_path = os.path.join(perturb_dir, cls_name, img_name)
    perturb_image = Image.open(perturb_image_path).convert('RGB')

    # 展示perturb图片
    # plt.imshow(perturb_image)
    # plt.show()

    # 进行的图片扩增
    trans_idx = random.randint(0, len(trans_list)-1)
    # trans_idx = 1
    cur_trans = trans_list[trans_idx]
    aug_image = cur_trans(plt_transformer(image[0]))
    aug_image_tensor = tensor_transformer(aug_image)

    # # 将perturb与org图片合并
    # perturb_and_org = transforms.ToTensor()(perturb_image) * (1 - plt_mask) + image[0] * plt_mask

    # 将perturb与扩增图片合并
    perturb_and_aug = transforms.ToTensor()(perturb_image) * (1 - plt_mask) + aug_image_tensor * plt_mask

    # print(f'image:{image.dtype}, perturb_and_org:{perturb_and_org.dtype}')

    # # 对结果进行对比
    # org_out = ped_model(image)
    # print(f'org:{torch.softmax(org_out, dim=1)}')
    # perturb_out = ped_model(perturb_and_aug.unsqueeze(0).float())
    # print(f'perturb:{torch.softmax(perturb_out, dim=1)}')

    # # 保存转换过的图片

    # 保存的名字
    # 保存perturb + aug图片
    new_image_name = os.path.splitext(img_name)[0] + '_' + trans_name_list[trans_idx] + os.path.splitext(img_name)[-1]
    save_path = os.path.join(save_dir, cls_name, new_image_name)
    save_image_tensor(perturb_and_aug.unsqueeze(0), new_image_name)


    # comb_image = transforms.ToTensor()(aug_image) * (1 - plt_mask) + image[0] * plt_mask
    # aug_comb_attack = transforms.ToTensor()(aug_image) * (1 - plt_mask) + image_attacked * plt_mask
    # save_image_tensor(comb_image.unsqueeze(0), save_path)

    # 保存原始图片
    save_path = os.path.join(save_dir, cls_name, img_name)
    save_image_tensor(image, save_path)


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


    # plt_imgs = 4
    #
    # # 创建一个包含两个子图的网格
    # plt.subplot(1, plt_imgs, 1)
    # plt.imshow(plt_transformer(image[0]))
    # plt.title('org')
    # plt.axis('off')  # 关闭坐标轴
    #
    # plt.subplot(1, plt_imgs, 2)
    # plt.imshow(plt_cam)
    # plt.title('cam')
    # plt.axis('off')  # 关闭坐标轴
    #
    # plt.subplot(1, plt_imgs, 3)
    # plt.imshow(plt_mask[0], cmap='gray')
    # # plt.imshow(plt_transformer(norm_cam[0]))
    # plt.title('mask')
    #
    # plt.subplot(1, plt_imgs, 4)
    # plt.imshow(plt_transformer(perturb_and_aug))
    # # plt.title('')
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














































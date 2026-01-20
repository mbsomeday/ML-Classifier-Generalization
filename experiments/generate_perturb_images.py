'''
    生成在图片部分区域进行perturb的图片
'''

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, torch
from torch import nn, optim
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchcam.methods.gradient import LayerCAM
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
import matplotlib.pyplot as plt

from data.dataset import my_dataset
from utils.utils import load_model, DEVICE, save_image_tensor



def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_name_list', default=['D1'])
    parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--save_dir', type=str, default=r'E:\Bias_Reduction_Summary\Datasets\Perturbations\D1_ECP')

    opts = parser.parse_args()

    return opts



def gen_part_aug(opts):
    # dataset
    train_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key='Stage6_org', txt_name='train.txt')
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=False)

    # model
    dataset_classifier = models.efficientnet_b0(weights=None, num_classes=3)
    dataset_classifier = load_model(dataset_classifier, opts.ds_weights_path)
    dataset_classifier = dataset_classifier.to(DEVICE).eval()

    # 选择CAM算法
    grad_layer = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8']
    layerCam_extractor = LayerCAM(dataset_classifier, target_layer=grad_layer)

    # transformers
    plt_transformer = transforms.ToPILImage()
    tensor_transformer = transforms.ToTensor()
    plt_resize = transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)

    # 对图片进行perturbation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dataset_classifier.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=dataset_classifier,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=2,
    )
    attack = FastGradientMethod(estimator=classifier, eps=0.1)

    # 循环遍历
    for data_dict in tqdm(train_loader):
        image = data_dict['image'].to(DEVICE)
        image_np = image.numpy()        # 用于生成perturb的图片

        ds_out = dataset_classifier(image)
        cam = layerCam_extractor(ds_out.squeeze(0).argmax().item(), ds_out)

        # 将不同glayer的cam合并
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

        plt_cam = plt_transformer(norm_cam[0])

        # 生成mask
        cam_array = np.array(plt_cam)
        cam_mask = cam_array < (0.4 * cam_array.max())  # 将图像不感兴趣的区域保留
        # cam_mask = cam_array >= (0.4 * cam_array.max())       # 将图像感兴趣的区域保留
        plt_mask = cam_mask * 1.0

        plt_mask = plt_mask[np.newaxis, :]

        # 生成perturbation图片
        perturb_image = attack.generate(x=image_np)




        break


def gen_perturbation_image(opts):
    # dataset
    train_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key='Stage6_org', txt_name='train.txt')
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=False)

    # model
    dataset_classifier = models.efficientnet_b0(weights=None, num_classes=3)
    dataset_classifier = load_model(dataset_classifier, opts.ds_weights_path)
    dataset_classifier = dataset_classifier.to(DEVICE).eval()

    # 对图片进行perturbation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dataset_classifier.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=dataset_classifier,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=3,
    )
    pertub_method = FastGradientMethod(estimator=classifier, eps=0.04)

    save_ped_dir = os.path.join(opts.save_dir, 'pedestrian')
    save_noPed_dir = os.path.join(opts.save_dir, 'nonPedestrian')
    if not os.path.exists(save_ped_dir):
        os.mkdir(save_ped_dir)
    if not os.path.exists(save_noPed_dir):
        os.mkdir(save_noPed_dir)

    # 循环遍历
    for data_dict in tqdm(train_loader):
        image = data_dict['image']  # tensor [n, 3, 224, 224]
        img_name = data_dict['img_name'][0]
        image_np = image.numpy()  # nparray [1, 3, 224, 224]

        img_label = int(data_dict['ped_label'][0])
        label = 'nonPedestrian' if img_label == 0 else 'pedestrian'

        # 生成perturb图片
        perturb_image = pertub_method.generate(x=image_np)
        perturb_image = perturb_image[0].transpose((1, 2, 0))
        perturb_tensor = torch.from_numpy(perturb_image).permute(2, 0, 1).unsqueeze(0)

        # 保存perturb图片
        save_path = os.path.join(opts.save_dir, label, img_name)
        save_image_tensor(input_tensor=perturb_tensor, filename=save_path)

        # # 结果对比
        # org_out = dataset_classifier(image)
        # print(f'\norg:{torch.softmax(org_out, dim=1)}')
        # att_out = dataset_classifier(perturb_tensor)
        # print(f'att:{torch.softmax(att_out, dim=1)}')
        #
        # # 展示图片
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(transforms.ToPILImage()(image[0]))
        # plt.title('Original Image')
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.clip(perturb_image, 0, 1))
        # plt.title('Perturbated Image')
        # plt.show()

        # break







if __name__ == '__main__':
    print('start')

    opts = get_opts()
    gen_perturbation_image(opts)






































'''
    两个功能：
        1. gen_perturbation_image()：生成full perturbated图片
        2. gen_part_aug()：将图片部分perturbated+aug
'''

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, torch, random
from torch import nn, optim
import torchvision.transforms.functional as F
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

    # 用于云端
    parser.add_argument('--task', type=str, choices=['perturbed', 'combined'], required=True, default=None)
    parser.add_argument('--ds_name_list', nargs='+')
    parser.add_argument('--txt_name', type=str)
    parser.add_argument('--num_classes', type=int, help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    parser.add_argument('--model_weights', type=str)

    parser.add_argument('--path_key', type=str)
    parser.add_argument('--batch_size', type=int, default=1)

    # 生成perturbation图片
    parser.add_argument('--perturb_save_dir', type=str)

    # 生成perturbation+aug图片
    parser.add_argument('--perturb_dir', type=str)
    parser.add_argument('--compondImg_save_dir', type=str)


    # 用于本地测试
    # parser.add_argument('--task', type=str, choices=['perturbed', 'cmbined'], default='cmbined')
    # parser.add_argument('--ds_name_list', nargs='+', default=['D2'])
    # parser.add_argument('--txt_name', type=str, default='test.txt')
    # parser.add_argument('--num_classes', type=int, default=3, help='the number is 3 when using dataset classifier, and is 2 when using pedestrian classifier')
    # parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    #
    # # parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\baselines\D2\efficientNetB0_D2_51_Baseline-19-2.00064.pth')
    #
    # # parser.add_argument('--ds_weights_path', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    # parser.add_argument('--path_key', type=str, default='Stage6_org')
    # parser.add_argument('--batch_size', type=int, default=1)
    #
    # # 生成perturbation图片
    # parser.add_argument('--perturb_save_dir', type=str, default=None)
    #
    # # 生成perturbation+aug图片
    # parser.add_argument('--perturb_dir', type=str, default=r'D:\my_phd\dataset\Stage6\stage6_citypersons\All_Processor\DSCAM\onlyPurturbations\test')
    # # parser.add_argument('--perturb_dir', type=str, default=None)
    # parser.add_argument('--compondImg_save_dir', type=str, default=None)

    opts = parser.parse_args()

    return opts




def gen_perturbation_image(opts):
    '''
        生成整张perturbated的图片
    '''

    # dataset
    train_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key='Stage6_org', txt_name=opts.txt_name)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=False)

    # model
    get_classifier = models.efficientnet_b0(weights=None, num_classes=opts.num_classes)
    get_classifier = load_model(get_classifier, opts.model_weights)
    get_classifier = get_classifier.to(DEVICE).eval()

    # 对图片进行perturbation
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(get_classifier.parameters(), lr=0.01)
    classifier = PyTorchClassifier(
        model=get_classifier,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=3,
    )
    pertub_method = FastGradientMethod(estimator=classifier, eps=0.04)

    save_ped_dir = os.path.join(opts.perturb_save_dir, 'pedestrian')
    save_noPed_dir = os.path.join(opts.perturb_save_dir, 'nonPedestrian')
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
        save_path = os.path.join(opts.perturb_save_dir, label, img_name)
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
        #
        # break

def gen_perturb_aug(opts):
    '''
        生成combined images
    '''

    # dataset
    train_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key=opts.path_key, txt_name=opts.txt_name)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=False)

    # model
    get_classifier = models.efficientnet_b0(weights=None, num_classes=opts.num_classes)
    get_classifier = load_model(get_classifier, opts.model_weights)
    get_classifier = get_classifier.to(DEVICE).eval()

    # 选择CAM算法
    grad_layer = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8']
    layerCam_extractor = LayerCAM(get_classifier, target_layer=grad_layer)

    # transformers
    plt_transformer = transforms.ToPILImage()
    tensor_transformer = transforms.ToTensor()
    plt_resize = transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)

    # augmentation
    def hflip(img):
        return F.hflip(img)

    def rotate(img):
        angle = random.randint(-10, 10)
        return F.rotate(img, angle)

    def jittor(img):
        color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
        return color_jitter(img)

    def gaussian(img):
        sigma = random.uniform(0.1, 1.0)
        img = F.gaussian_blur(img, kernel_size=[5, 5], sigma=[sigma, sigma])
        return img

    aug_list = [hflip, rotate, jittor, gaussian]
    aug_name_list = ['Hflip', 'Rotate', 'Jittor', 'Gaussian']

    # 循环遍历
    for data_dict in tqdm(train_loader):
        image = data_dict['image'].to(DEVICE)
        img_name = data_dict['img_name'][0]
        image_path = data_dict['img_path'][0]
        cls_name = image_path.split(os.sep)[-2]

        out = get_classifier(image)
        cam = layerCam_extractor(out.squeeze(0).argmax().item(), out)

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
        norm_cam = (comb_layercam - cam_min) / (((cam_max - cam_min) + 1e-08))

        plt_cam = plt_transformer(norm_cam[0])

        # 生成mask
        cam_array = np.array(plt_cam)
        cam_mask = cam_array < (0.5 * cam_array.max())        # 值高的部分为黑色
        # cam_mask = cam_array >= (0.5 * cam_array.max())         # 值低的部分为黑色
        plt_mask = cam_mask * 1.0

        plt_mask = plt_mask[np.newaxis, :]
        plt_mask = torch.from_numpy(plt_mask).float().to(DEVICE)

        plt_mask = 1 - plt_mask

        # 进行的图片扩增
        # 这里要注意，如果是flip和rotate，需要先将perturbation与org image结合，然后再进行aug操作，否则，CAM对应的位置会变
        random_aug_id = random.randint(0, len(aug_list) - 1)
        cur_aug_operation = aug_list[random_aug_id]

        # random_aug_id = 2       # 仅用于测试
        # cur_aug_operation = aug_list[random_aug_id] # 仅用于测试
        # cur_aug_operation = transforms.Grayscale(num_output_channels=3) # 仅用于测试

        # ---------- 组合augmentation代码 开始 ----------

        # 加载perturb图片
        perturb_image_path = os.path.join(opts.perturb_dir, cls_name, img_name)
        perturb_image = Image.open(perturb_image_path).convert('RGB')

        if random_aug_id == 0 or random_aug_id == 1:
            original_and_org = cur_aug_operation(image[0])
        else:
            aug_image = cur_aug_operation(plt_transformer(image[0]))
            aug_image_tensor = tensor_transformer(aug_image).to(DEVICE)
            original_and_org = aug_image_tensor * (1 - plt_mask) + image[0] * plt_mask

            # # 测试用代码
            # perturb_and_aug = tensor_transformer(perturb_image).to(DEVICE) * (1 - plt_mask) + aug_image_tensor * plt_mask
            # aug_and_perturb = tensor_transformer(perturb_image).to(DEVICE) * plt_mask + aug_image_tensor * (1-plt_mask)
            # Aug_and_org = aug_image_tensor * (1 - plt_mask) + image[0] * plt_mask
            # org_and_aug = aug_image_tensor * plt_mask + image[0] * (1-plt_mask)

        # save_AugOrg_name = os.path.splitext(img_name)[0] + '_' + aug_name_list[random_aug_id] + 'Org' + os.path.splitext(img_name)[-1]       # AugOrg
        save_AugOrg_name = os.path.splitext(img_name)[0] + '_Org' + aug_name_list[random_aug_id] + os.path.splitext(img_name)[-1]        # OrgAug
        save_AugOrg_path = os.path.join(opts.compondImg_save_dir, cls_name, save_AugOrg_name)
        save_image_tensor(original_and_org.unsqueeze(0), save_AugOrg_path)

        # ---------- 组合augmentation代码 结束 ----------


        # # ---------- 需要用到perturb部分的代码 开始 ----------

        # # 加载perturb图片
        # perturb_image_path = os.path.join(opts.perturb_dir, cls_name, img_name)
        # perturb_image = Image.open(perturb_image_path).convert('RGB')
        #
        # # 不论mask如何，perturb都对应mask的黑色部分，aug对应mask的白色部分
        # # aug操作为flip和rotate，
        # if random_aug_id == 0 or random_aug_id == 1:
        #     perturb_and_org = tensor_transformer(perturb_image).to(DEVICE) * (1 - plt_mask) + image[0] * plt_mask
        #     perturb_and_aug = cur_aug_operation(perturb_and_org)
        # else:
        #     # 直接将perturb与org图片合并
        #     aug_image = cur_aug_operation(plt_transformer(image[0]))
        #     aug_image_tensor = tensor_transformer(aug_image).to(DEVICE)
        #     perturb_and_aug = tensor_transformer(perturb_image).to(DEVICE) * (1 - plt_mask) + aug_image_tensor * plt_mask
        #
        # # save_perturbAug_name = os.path.splitext(img_name)[0] + '_perturb' + aug_name_list[random_aug_id] + os.path.splitext(img_name)[-1]     # PerturbAug
        # save_perturbAug_name = os.path.splitext(img_name)[0] + '_' + aug_name_list[random_aug_id] + 'perturb' + os.path.splitext(img_name)[-1]       # AugPerturb
        # save_perturbAug_path = os.path.join(opts.compondImg_save_dir, cls_name, save_perturbAug_name)
        # save_image_tensor(perturb_and_aug.unsqueeze(0), save_perturbAug_path)

        # ---------- 需要用到perturb部分的代码 结束 ----------


        # print(f'save_perturbAug_name:{save_perturbAug_name}')
        # print(f'save_perturbAug_path:{save_perturbAug_path}')


        # plt.imshow(plt_cam)
        # plt.axis('off')  # 关闭坐标轴
        # plt.show()

        # 创建一个包含两个子图的网格

        # plt_imgs = 5
        #
        # plt.subplot(1, plt_imgs, 1)
        # plt.imshow(plt_transformer(image[0]))
        # plt.title('org')
        # plt.axis('off')  # 关闭坐标轴
        #
        # plt.subplot(1, plt_imgs, 2)
        # plt.imshow(plt_cam)
        # plt.title('cam')
        # plt.axis('off')  # 关闭坐标轴

        # plt.subplot(1, plt_imgs, 3)
        # plt.imshow(plt_mask[0], cmap='gray')
        # # plt.imshow(plt_transformer(norm_cam[0]))
        # plt.title('mask')

        # plt.subplot(1, plt_imgs, 4)
        # plt.imshow(plt_transformer(aug_image_tensor))
        # plt.title('aug')

        # plt.subplot(1, plt_imgs, 5)
        # plt.imshow(plt_transformer(original_and_org))
        # title = f'{aug_name_list[random_aug_id]} + Perturb'
        # plt.title(title)
        # plt.axis('off')  # 关闭坐标轴
        #
        # # 显示图片
        # plt.show()
        #
        # break






if __name__ == '__main__':
    print('start')

    opts = get_opts()

    for k, v in vars(opts).items():
        print(f'{k}: {v}')

    if opts.task == 'perturbed':
        gen_perturbation_image(opts)
    else:
        gen_perturb_aug(opts)

    # # 生成perturbation图片
    # gen_perturbation_image(opts)

    # 将perturbation与aug结合生成新图片
    # gen_perturb_aug(opts)









































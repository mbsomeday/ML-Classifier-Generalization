'''
    1. to generate operated datasets in each group
    2. to generate the corresponding txt files
'''
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

import argparse, torch, random, cv2
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
import torch.nn.functional as nnF

from data.dataset import my_dataset
from utils.utils import load_model, DEVICE, save_image_tensor



class Random_Aug():
    def __init__(self):
        self.aug_list = [self.hflip, self.rotate, self.jittor, self.gaussian]
        self.aug_name_list = ['Hflip', 'Rotate', 'Jittor', 'Gaussian']
        self.random_aug_id = -1

    def __call__(self):
        self.random_aug_id = random.randint(0, len(self.aug_list) - 1)
        # self.random_aug_id = 2        # for test
        cur_aug_operation = self.aug_list[self.random_aug_id]
        return cur_aug_operation

    def hflip(self, img):
        return transforms.RandomHorizontalFlip(p=1.0)(img)

    def rotate(self, img):
        # allow tensor and PIL type image
        angle = random.randint(-10, 10)
        return transforms.RandomRotation(degrees=(angle, angle))(img)

    def jittor(self, img):
        color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
        return color_jitter(img)

    def gaussian(self, img):
        sigma = random.uniform(0.1, 1.0)
        return transforms.GaussianBlur(kernel_size=5, sigma=sigma)(img)

def create_dirs(base_dir, task_name, txt_name, exist_ok=False):
    '''
        创建存储图片的文件夹
        自动创建 行人 与 非行人 的文件夹
    '''
    ped_dir_path = os.path.join(base_dir, task_name, txt_name, 'pedestrian')
    nonPed_dir_path = os.path.join(base_dir, task_name, txt_name, 'nonPedestrian')

    if exist_ok is True and (os.path.exists(ped_dir_path) and os.path.exists(nonPed_dir_path)):
        print(f'{ped_dir_path} exists, not create.')
        print(f'{nonPed_dir_path} exists, not create.')
    else:
        try:
            os.makedirs(ped_dir_path, exist_ok=exist_ok)
            os.makedirs(nonPed_dir_path, exist_ok=exist_ok)

        except FileExistsError:
            raise FileExistsError(f'File already exists. {ped_dir_path} and {nonPed_dir_path}')

        print(f'Create dir {ped_dir_path}')
        print(f'Create dir {nonPed_dir_path}')


def gen_perturbation_image(opts):
    '''
        to generate perturbed images with Fast Gradient
    '''

    for txt_name in opts.txt_name_list:
        # dataset
        get_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key=opts.path_key, txt_name=txt_name)
        get_loader = DataLoader(get_dataset, batch_size=opts.batch_size, shuffle=False)

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

        # ---------- create saving dir ----------
        create_dirs(opts.genImg_save_dir, task_name='OnlyPerturb', txt_name=txt_name.split('.')[0], exist_ok=opts.dir_exist_ok)


        # 循环遍历
        for idx, data_dict in enumerate(tqdm(get_loader)):
            image = data_dict['image']  # tensor [n, 3, 224, 224]
            img_name = data_dict['img_name'][0]
            image_np = image.numpy()  # nparray [1, 3, 224, 224]
            image_path = data_dict['img_path'][0]
            cls_name = image_path.split(os.sep)[-2]

            # 生成perturb图片
            perturb_image = pertub_method.generate(x=image_np)
            perturb_image = perturb_image[0].transpose((1, 2, 0))
            perturb_tensor = torch.from_numpy(perturb_image).permute(2, 0, 1).unsqueeze(0)

            # 保存perturb图片
            save_path = os.path.join(opts.genImg_save_dir, 'OnlyPerturb', txt_name.split('.')[0], cls_name, img_name)
            save_image_tensor(input_tensor=perturb_tensor, filename=save_path)

            # if idx == 6:
            #     break

        print('-' * 10, f'Finished {txt_name}', '-' * 10)

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


def gen_CAM_Mask(opts):
    '''
        to generate LayerCAM and mask(threshold=0.5)
    '''

    # CAM_save_dir = os.path.join(opts.genImg_save_dir, 'CAM')
    # Mask_save_dir = os.path.join(opts.genImg_save_dir, 'Mask')

    for txt_name in opts.txt_name_list:
        # ---------- create saving dir ----------
        create_dirs(opts.genImg_save_dir, task_name='CAM', txt_name=txt_name.split('.')[0], exist_ok=opts.dir_exist_ok)
        create_dirs(opts.genImg_save_dir, task_name='Mask', txt_name=txt_name.split('.')[0], exist_ok=opts.dir_exist_ok)

        # dataset
        get_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key=opts.path_key, txt_name=txt_name)
        get_loader = DataLoader(get_dataset, batch_size=opts.batch_size, shuffle=False)

        # model
        get_classifier = models.efficientnet_b0(weights=None, num_classes=opts.num_classes)
        get_classifier = load_model(get_classifier, opts.model_weights)
        get_classifier = get_classifier.to(DEVICE).eval()

        # 选择CAM算法
        grad_layer = ['features.0', 'features.1', 'features.2', 'features.3', 'features.4', 'features.5', 'features.6', 'features.7', 'features.8']
        layerCam_extractor = LayerCAM(get_classifier, target_layer=grad_layer)

        # # transformers
        # plt_transformer = transforms.ToPILImage()
        # tensor_transformer = transforms.ToTensor()
        # plt_resize = transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)

        for idx, data_dict in enumerate(tqdm(get_loader)):
            image = data_dict['image'].to(DEVICE)
            img_name = data_dict['img_name'][0]
            image_path = data_dict['img_path'][0]
            cls_name = image_path.split(os.sep)[-2]

            out = get_classifier(image)
            cam_list = layerCam_extractor(out.squeeze(0).argmax().item(), out)
            print(f'cam_list:{len(cam_list)}')

            # # 将不同glayer的cam合并
            # resized_hp = []
            # for hp in cam_list:
            #     hp = plt_transformer(hp)
            #     hp = hp.resize((224, 224), resample=Image.BICUBIC)
            #     cur_hp = tensor_transformer(hp).unsqueeze(0)
            #     resized_hp.append(cur_hp)
            #
            # vis_heatmaps = torch.cat(resized_hp, dim=0)
            # comb_layercam = torch.sum(vis_heatmaps, 0).unsqueeze(0)
            # (cam_min, cam_max) = (comb_layercam.min(), comb_layercam.max())
            # norm_cam = (comb_layercam - cam_min) / (((cam_max - cam_min) + 1e-08))

            # 将不同layer的cam进行resize
            # ---------- gen & save CAM ----------
            resized_cam_list = []
            for cur_cam in cam_list:
                cur_cam = cur_cam.unsqueeze(0)
                resized = nnF.interpolate(cur_cam, size=(224, 224), mode='bilinear', align_corners=False)
                resized_cam_list.append(resized)

            cams_tensor = torch.cat(resized_cam_list, dim=0)
            sum_cam = torch.sum(cams_tensor, dim=0)

            cam_min, cam_max = sum_cam.min(), sum_cam.max()
            norm_cam = (sum_cam - cam_min) / (((cam_max - cam_min) + 1e-08))
            cam_np = norm_cam.squeeze().detach().cpu().numpy()  # [224, 224]
            cam_uint8 = np.uint8(255 * cam_np) # 转成 0-255 uint8
            color_cam = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            color_cam = cv2.cvtColor(color_cam, cv2.COLOR_BGR2RGB)  # OpenCV 是 BGR，需要转成 RGB

            cam_save_path = os.path.join(opts.genImg_save_dir, 'CAM', txt_name.split('.')[0], cls_name, img_name)
            Image.fromarray(color_cam).save(cam_save_path)
            # print(f'cam_save_path:{cam_save_path}')

            # ---------- gen & save mask ----------
            # plt_cam = plt_transformer(norm_cam[0])
            # cam_array = np.array(plt_cam)

            cam_array = np.array(cam_np)
            threshold = 0.5
            cam_mask = cam_array < threshold  # 这里直接跟threshold比，因为cam_array已经归一化了，其max为1.0，值高的部分为黑色
            # cam_mask = cam_array >= (threshold * cam_array.max())         # 值低的部分为黑色
            plt_mask = cam_mask * 1.0

            mask_uint8 = np.uint8(plt_mask * 255)       # 转成 0/255

            mask_save_path = os.path.join(opts.genImg_save_dir, 'Mask', txt_name.split('.')[0], cls_name, img_name)
            Image.fromarray(mask_uint8).save(mask_save_path)
            # print(f'mask_save_path:{mask_save_path}')

            # # 将不同glayer的cam合并
            # resized_hp = []
            # for hp in cam:
            #     print('cam', type(hp), hp.shape)
            #     hp = plt_transformer(hp)
            #     hp = hp.resize((224, 224), resample=Image.BICUBIC)
            #     cur_hp = tensor_transformer(hp).unsqueeze(0)
            #     resized_hp.append(cur_hp)
            #
            # vis_heatmaps = torch.cat(resized_hp, dim=0)
            # comb_layercam = torch.sum(vis_heatmaps, 0).unsqueeze(0)
            # (cam_min, cam_max) = (comb_layercam.min(), comb_layercam.max())
            # norm_cam = (comb_layercam - cam_min) / (((cam_max - cam_min) + 1e-08))
            #
            # cam_np = norm_cam.squeeze().detach().cpu().numpy()  # [224, 224]
            # cam_uint8 = np.uint8(255 * cam_np) # 转成 0-255 uint8
            # color_cam = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            # color_cam = cv2.cvtColor(color_cam, cv2.COLOR_BGR2RGB)  # OpenCV 是 BGR，需要转成 RGB

            # save_path = os.path.join(save_dir, img_name)
            # Image.fromarray(color_cam).save('out.jpg')

            # if idx == 6:
            #     break


def gen_PerturbAug_AugPerturb(opts):

    # 先确定已经有onlyPerturb和mask
    onlyPerturb_dir = os.path.join(opts.genImg_save_dir, 'OnlyPerturb')
    mask_dir = os.path.join(opts.genImg_save_dir, 'Mask')
    if not (os.path.exists(onlyPerturb_dir) and os.path.exists(mask_dir)):
        raise FileNotFoundError('Should generate perturb and CAM first!')
    operation_list = ['PerturbAug', 'AugPerturb']
    def PerturbAug_AugPerturb_operation():
        # 仅用于测试，变为灰度图
        cur_aug_operation = random_aug_operation()
        # cur_aug_operation = transforms.Grayscale(num_output_channels=3)

        if random_aug_operation.random_aug_id == 0 or random_aug_operation.random_aug_id == 1:
            operated_image = tensor_transformer(onlyPerturb_image).to(DEVICE) * (1 - mask_image) + image * mask_image
            operated_image = cur_aug_operation(operated_image)

            # PerturbAug_image = tensor_transformer(onlyPerturb_image).to(DEVICE) * (1 - mask_image) + image * mask_image
            # PerturbAug_image = cur_aug_operation(PerturbAug_image)

        else:
            # 直接将perturb与org图片合并
            # aug_image = cur_aug_operation(plt_transformer(image[0]))
            aug_image_tensor = cur_aug_operation(image)
            operated_image = tensor_transformer(onlyPerturb_image).to(DEVICE) * (1 - mask_image) + aug_image_tensor * mask_image

        return operated_image


    # ---------- augmentation class ----------
    random_aug_operation = Random_Aug()

    # transformers
    plt_transformer = transforms.ToPILImage()
    tensor_transformer = transforms.ToTensor()

    # 读取onlyPerturb和mask
    for txt_name in opts.txt_name_list:
        # ---------- create saving dirs ----------
        create_dirs(base_dir=opts.genImg_save_dir, task_name='PerturbAug', txt_name=txt_name.split('.')[0], exist_ok=opts.dir_exist_ok)
        create_dirs(base_dir=opts.genImg_save_dir, task_name='AugPerturb', txt_name=txt_name.split('.')[0], exist_ok=opts.dir_exist_ok)

        # dataset
        get_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key=opts.path_key, txt_name=txt_name)
        get_loader = DataLoader(get_dataset, batch_size=opts.batch_size, shuffle=False)

        # perturb and masks
        onlyPerturb_dir = os.path.join(opts.genImg_save_dir, 'OnlyPerturb', txt_name.split('.')[0])
        mask_dir = os.path.join(opts.genImg_save_dir, 'Mask', txt_name.split('.')[0])

        for idx, data_dict in enumerate(tqdm(get_loader)):
            image = data_dict['image'].to(DEVICE)[0]
            img_name = data_dict['img_name'][0]
            image_path = data_dict['img_path'][0]
            cls_name = image_path.split(os.sep)[-2]

            onlyPerturb_image_dir = os.path.join(onlyPerturb_dir, cls_name, img_name)
            mask_image_path = os.path.join(mask_dir, cls_name, img_name)

            onlyPerturb_image = Image.open(onlyPerturb_image_dir).convert('RGB')
            mask_image = Image.open(mask_image_path).convert("L")
            mask_image = tensor_transformer(mask_image).to(DEVICE)
            # mask_image = np.array(mask_image)
            # mask_image = (mask_image > 0).astype(np.uint8)
            mask_image = (mask_image > 0).float()

            for opt_idx, opt in enumerate(operation_list):
                if opt == 'AugPerturb':
                    mask_image = 1 - mask_image
                operated_image = PerturbAug_AugPerturb_operation()
                save_path = os.path.join(opts.genImg_save_dir, operation_list[opt_idx], txt_name.split('.')[0], cls_name, img_name)
                save_image_tensor(input_tensor=operated_image, filename=save_path)


            # if idx == 6:
            #     break


def gen_AugOrg_OrgAug(opts):
    # 先确定已经有mask
    mask_dir = os.path.join(opts.genImg_save_dir, 'Mask')
    if not os.path.exists(mask_dir):
        raise FileNotFoundError('Should generate CAM mask first!')

    operation_list = ['AugOrg', 'OrgAug']

    def AugOrg_OrgAug_operation():
        cur_aug_operation = random_aug_operation()
        # cur_aug_operation = transforms.Grayscale(num_output_channels=3)  # 仅用于测试

        if random_aug_operation.random_aug_id == 0 or random_aug_operation.random_aug_id == 1:
            operated_image = cur_aug_operation(image)

        else:
            # 直接将Aug与org图片合并
            aug_image_tensor = cur_aug_operation(image).to(DEVICE)
            operated_image = aug_image_tensor * (1 - mask_image) + image * mask_image

        return operated_image

    # ---------- augmentation class ----------
    random_aug_operation = Random_Aug()

    tensor_transformer = transforms.ToTensor()

    # 读取onlyPerturb和mask
    for txt_name in opts.txt_name_list:
        # ---------- create saving dirs ----------
        create_dirs(base_dir=opts.genImg_save_dir, task_name='AugOrg', txt_name=txt_name.split('.')[0], exist_ok=opts.dir_exist_ok)
        create_dirs(base_dir=opts.genImg_save_dir, task_name='OrgAug', txt_name=txt_name.split('.')[0], exist_ok=opts.dir_exist_ok)

        # ---------- dataset ----------
        get_dataset = my_dataset(ds_name_list=opts.ds_name_list, path_key=opts.path_key, txt_name=txt_name)
        get_loader = DataLoader(get_dataset, batch_size=opts.batch_size, shuffle=False)

        # ---------- get Mask image dir ----------
        mask_dir = os.path.join(opts.genImg_save_dir, 'Mask', txt_name.split('.')[0])

        for idx, data_dict in enumerate(tqdm(get_loader)):
            image = data_dict['image'].to(DEVICE)[0]
            img_name = data_dict['img_name'][0]
            image_path = data_dict['img_path'][0]
            cls_name = image_path.split(os.sep)[-2]

            mask_image_path = os.path.join(mask_dir, cls_name, img_name)
            mask_image = Image.open(mask_image_path).convert("L")
            mask_image = tensor_transformer(mask_image).to(DEVICE)
            # mask_image = np.array(mask_image)
            # mask_image = (mask_image > 0).astype(np.uint8)
            mask_image = (mask_image > 0).float()

            for opt_idx, opt in enumerate(operation_list):
                if opt == 'OrgAug':
                    mask_image = 1 - mask_image
                operated_img = AugOrg_OrgAug_operation()
                save_path = os.path.join(opts.genImg_save_dir, str(opt), txt_name.split('.')[0], cls_name, img_name)
                save_image_tensor(input_tensor=operated_img, filename=save_path)

            # if idx == 6:
            #     break



def gen_txt(opts):
    '''
        生成相应的txt文件
    '''
    txt_save_path = os.path.join(opts.txt_save_dir, opts.txt_save_name)
    # 先保存org txt中的信息
    with open(opts.add_org_txt, 'r') as f:
        org_txt_info = f.readlines()

    with open(txt_save_path, 'a') as f:
        for item in org_txt_info:
            f.write(item)

    # 再保存operated data info

    operated_image_path = os.path.join(opts.image_base_dir, opts.imgae_group_path)
    msg_list = []

    for cls_name in os.listdir(operated_image_path):
        if cls_name == 'nonPedestrian':
            cur_label = ' 0'
        elif cls_name == 'pedestrian':
            cur_label = ' 1'
        else:
            raise ValueError(f'Unknown class name {cls_name}')

        operated_images = os.listdir(os.path.join(operated_image_path, cls_name))
        for image_path in operated_images:
            msg = os.path.join(opts.imgae_group_path, cls_name, image_path) + cur_label + '\n'
            msg_list.append(msg)

    with open(txt_save_path, 'a') as f:
        for item in msg_list:
            f.write(item)

    print(f'Save to {txt_save_path}.')





GEN_FUNC_REGISTRY = {
    'OnlyPerturb': gen_perturbation_image,
    'CAM_Mask': gen_CAM_Mask,
    'Aug_Perturb': gen_PerturbAug_AugPerturb,
    'Aug_Org': gen_AugOrg_OrgAug,
}



























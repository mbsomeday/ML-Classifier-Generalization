import os, random
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from utils.utils import save_image_tensor


class random_aimage_aug():
    '''
        将augmentation保存到save_dir中
    '''
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.image_list = self.get_images()

        self.save_dir = os.path.join(self.base_dir, 'augmentation_train')
        self.ped_dir = os.path.join(self.save_dir, 'pedestrian')
        self.nonPed_dir = os.path.join(self.save_dir, 'nonPedestrian')
        if not os.path.exists(self.ped_dir):
            os.makedirs(self.ped_dir)
        if not os.path.exists(self.nonPed_dir):
            os.makedirs(self.nonPed_dir)

        self.aug_list = [self.hflip, self.rotate, self.jittor, self.gaussian]
        self.aug_name = ['hflip', 'rotate', 'jittor', 'gaussian']
        self.num_list = list(range(len(self.aug_list)))

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_images(self):
        with open(os.path.join(self.base_dir, 'dataset_txt', 'train.txt'), 'r') as f:
            data = f.readlines()

        image_list = []
        for item in data:
            item = os.path.join(self.base_dir, item.strip().split()[0])
            image_list.append(item)

        return image_list


    def hflip(self, img):
        return F.hflip(img)

    def rotate(self, img):
        angle = random.randint(-10, 10)
        return F.rotate(img, angle)

    def jittor(self, img):
        color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        )
        return color_jitter(img)

    def gaussian(self, img):
        sigma = random.uniform(0.1, 1.0)
        img = F.gaussian_blur(img, kernel_size=[5, 5], sigma=[sigma, sigma])
        return img


    def __call__(self):

        for image_path in tqdm(self.image_list):
            # print(data_dict.keys())

            # 随机选择一种augmentation
            rand_num = random.choice(self.num_list)
            aug_method = self.aug_list[rand_num]
            aug_name = self.aug_name[rand_num]

            # 读取原始图片
            org_image = Image.open(image_path).convert('RGB')
            org_image = self.img_transforms(org_image).unsqueeze(0)
            org_img_name = os.path.basename(image_path)
            cls_name = image_path.split(os.sep)[-2]

            # org_image = data_dict['image']
            # org_img_name = data_dict['img_name'][0]
            # org_img_path = data_dict['img_path'][0]
            # cls_name = org_img_path.split(os.sep)[-2]

            # 保存原始图片
            save_org_path = os.path.join(self.save_dir, cls_name, org_img_name)
            save_image_tensor(org_image, save_org_path)

            # 保存 aug 图片
            save_aug_name = os.path.splitext(org_img_name)[0] + '_' + aug_name + os.path.splitext(org_img_name)[-1]
            save_aug_path = os.path.join(self.save_dir, cls_name, save_aug_name)
            aug_image = aug_method(org_image)
            save_image_tensor(aug_image, save_aug_path)

            # break

    def gen_aug_txt(self):
        aug_txt_path = os.path.join(self.base_dir, 'dataset_txt', 'augmentation_train.txt')
        ped_list = os.listdir(self.ped_dir)
        nonPed_list = os.listdir(self.nonPed_dir)
        with open(aug_txt_path, 'a') as f:
            for item in ped_list:
                item = os.path.join('augmentation_train', 'pedestrian', item) + ' 1\n'
                f.write(item)
            for item in nonPed_list:
                item = os.path.join('augmentation_train', 'nonPedestrian', item) + ' 0\n'
                f.write(item)

































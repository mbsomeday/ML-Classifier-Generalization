import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

from configs.ds_path import PATHS


class my_dataset(Dataset):
    def __init__(self, ds_name_list, path_key, txt_name, ds_labels=None):
        '''
        :param ds_name_list:
        :param path_key: org_dataset
        :param ds_labels: 用于打乱数据集标签
        '''
        self.ds_name_list = ds_name_list
        self.ds_label_list = []
        self.path_key = path_key

        # 用于测试打乱dataset name和label的对应实验
        if ds_labels is None:
            for ds_name in ds_name_list:
                self.ds_label_list.append(int(ds_name[1]) - 1)
        else:
            self.ds_label_list = ds_labels
        if len(self.ds_name_list) != len(self.ds_label_list):
            raise ValueError(f'Dataset loader name and label match error. Names:{self.ds_name_list}, labels:{self.ds_label_list}')

        print(f'Mapping dataset names and labels: {ds_name_list} - {self.ds_label_list}')

        self.txt_name = txt_name
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.images, self.ped_labels, self.ds_labels = self.init_ImagesLabels()
        print(f'Get dataset: {ds_name_list}, txt_name: {txt_name}, total {len(self.images)} images')

    def init_ImagesLabels(self):
        images, ped_labels, ds_labels = [], [], []

        for ds_idx, ds_name in enumerate(self.ds_name_list):
            ds_label = self.ds_label_list[ds_idx]
            ds_dir = PATHS[self.path_key][ds_name]
            txt_path = os.path.join(ds_dir, 'dataset_txt', self.txt_name)

            print(f'Lodaing {txt_path}')

            with open(txt_path, 'r') as f:
                data = f.readlines()

            for data_idx, line in enumerate(data):
                line = line.replace('\\', os.sep)
                line = line.strip()
                contents = line.split()

                image_path = os.path.join(ds_dir, contents[0])
                images.append(image_path)
                ped_labels.append(contents[-1])
                ds_labels.append(ds_label)

        return images, ped_labels, ds_labels

    def get_ped_cls_num(self):
        '''
            获取行人和非行人类别的数量
        '''
        nonPed_num, ped_num = 0, 0
        for item in self.ped_labels:
            if item == '0':
                nonPed_num += 1
            elif item == '1':
                ped_num += 1
        return nonPed_num, ped_num

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ped_label = self.ped_labels[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)
        ped_label = np.array(ped_label).astype(np.int64)
        ds_label = np.array(ds_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'img_name': image_name,
            'img_path': image_path,
            'ped_label': ped_label,
            'ds_label': ds_label
        }

        return image_dict


class read_from_dir(Dataset):
    def __init__(self, base_dir, ds_label):
        '''
            base_dir下为pedestrian和nonPedestrian
        '''
        super().__init__()
        self.base_dir = base_dir

        self.labels = []
        self.images = []
        self.ds_labels = []
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        cls_dir = os.listdir(self.base_dir)

        self.nonPed_num, self.ped_num = 0, 0

        for cls in cls_dir:

            if cls == 'pedestrian':
                cur_label = 1
                self.ped_num += 1
            elif cls == 'nonPedestrian':
                cur_label = 0
                self.nonPed_num += 1
            else:
                raise ValueError('Wrong class name')

            image_list = os.listdir(os.path.join(self.base_dir, cls))
            for img in image_list:
                self.ds_labels.append(int(ds_label))
                self.labels.append(cur_label)
                self.images.append(os.path.join(self.base_dir, cls, img))

    def __len__(self):
        return len(self.images)

    def get_ped_cls_num(self):
        '''
            获取行人和非行人类别的数量
        '''

        return self.nonPed_num, self.ped_num

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ped_label = self.labels[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)
        ped_label = np.array(ped_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'img_name': image_name,
            'img_path': image_path,
            'ped_label': ped_label,
            'ds_label': ds_label
        }

        return image_dict


class read_from_dirs(Dataset):
    def __init__(self, ds_dir_list, ds_label_list):
        '''
            base_dir下为pedestrian和nonPedestrian
        '''
        super().__init__()
        assert len(ds_dir_list) == len(ds_label_list), "error: len(ds_dir_list) != len(ds_label_list)"

        self.ds_dir_list = ds_dir_list

        self.labels = []
        self.images = []
        self.ds_labels = []
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.nonPed_num, self.ped_num = 0, 0

        for idx, ds_dir in enumerate(ds_dir_list):
            print(ds_dir)
            cls_dir = os.listdir(ds_dir)
            for cls in cls_dir:
                if cls == 'pedestrian':
                    cur_label = 1
                    self.ped_num += 1
                elif cls == 'nonPedestrian':
                    cur_label = 0
                    self.nonPed_num += 1
                else:
                    raise ValueError('Wrong class name')

                image_list = os.listdir(os.path.join(ds_dir, cls))
                for img in image_list:
                    self.ds_labels.append(int(ds_label_list[idx]))
                    self.labels.append(cur_label)
                    self.images.append(os.path.join(ds_dir, cls, img))

        # for cls in cls_dir:
        #
        #     if cls == 'pedestrian':
        #         cur_label = 1
        #         self.ped_num += 1
        #     elif cls == 'nonPedestrian':
        #         cur_label = 0
        #         self.nonPed_num += 1
        #     else:
        #         raise ValueError('Wrong class name')
        #
        #     image_list = os.listdir(os.path.join(self.base_dir, cls))
        #     for img in image_list:
        #         self.ds_labels.append(int(ds_label))
        #         self.labels.append(cur_label)
        #         self.images.append(os.path.join(self.base_dir, cls, img))

    def __len__(self):
        return len(self.images)

    def get_ped_cls_num(self):
        '''
            获取行人和非行人类别的数量
        '''

        return self.nonPed_num, self.ped_num

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ped_label = self.labels[idx]
        ds_label = self.ds_labels[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.img_transforms(image)
        ped_label = np.array(ped_label).astype(np.int64)

        image_name = image_path.split(os.sep)[-1]

        image_dict = {
            'image': image,
            'img_name': image_name,
            'img_path': image_path,
            'ped_label': ped_label,
            'ds_label': ds_label
        }

        return image_dict



if __name__ == '__main__':
    print('test')

    base_dir_list = [
        r'E:\Bias_Reduction_Summary\Datasets\Perturbations\D1_perturb',
        r'E:\Bias_Reduction_Summary\Datasets\Perturbations\D2_perturb'
    ]
    ds_label_list = [0, 1]
    ds = read_from_dirs(base_dir_list, ds_label_list)
    print(len(ds))






























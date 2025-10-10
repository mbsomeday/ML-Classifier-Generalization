from abc import ABCMeta, abstractmethod
from json_stream import load
import ijson, os, random, json
from PIL import Image
import numpy as np
from tqdm import tqdm


class Base_Datasets():
    def __init__(self, base_dir):
        # pedestrian cropping size
        self.ped_min_size = 50
        self.ped_max_size = 224

    @abstractmethod
    def get_imgAndlabels(self, set_name):
        pass

    @abstractmethod
    def get_anno(self, json_path, image_path):
        pass

    @staticmethod
    def calc_crop(a, b, max_val):
        '''
            this function regulate the cropping size, aiming to put the pedestrian in the crop image center
        '''
        diff = b - a
        if diff == 224:  # the cropping size is 224
            return a, b
        elif diff < 224:
            if diff % 2 == 1:
                if a > 0:
                    a = a - 1
                else:
                    b = b + 1
            diff = b - a
            if diff % 2 == 0:
                change_size = (224 - diff) / 2
                if a < change_size:
                    b = b + change_size + (change_size - a)
                    a = 0
                elif (b + change_size) > max_val:
                    a = a - change_size - (b + change_size - max_val)
                    b = max_val
                else:
                    a = a - change_size
                    b = b + change_size
            return Base_Datasets.calc_crop(a, b, max_val)
        else:
            return -1, -1

    @staticmethod
    def calculate_iou(bbox1, bbox2):
        """
        计算两个边界框之间的IoU(Intersection over Union)

        参数:
            bbox1: [x1, y1, x2, y2] 左上和右下坐标
            bbox2: [x1, y1, x2, y2] 左上和右下坐标

        返回:
            iou值 (float)
        """
        # 确定相交区域的坐标
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        # 如果没有相交区域，则IoU为0
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # 计算相交区域面积
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # 计算两个边界框各自的面积
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # 计算并集面积
        union_area = bbox1_area + bbox2_area - intersection_area

        # 计算IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0

        return iou

    @staticmethod
    def filter_bboxes_by_iou(bbox_list, oobjId_list, iou_threshold=0.5):
        """
        过滤边界框，返回与所有其他边界框IoU都小于阈值的边界框

        参数:
            bbox_list: 边界框列表，每个边界框格式为[x1, y1, x2, y2]
            iou_threshold: IoU阈值，默认为0.5

        返回:
            过滤后的边界框列表
        """
        filtered_bboxes = []
        filtered_id = []
        n = len(bbox_list)

        for i in range(n):
            current_bbox = bbox_list[i]
            meets_criteria = True

            for j in range(n):
                if i == j:
                    continue  # 不与自己比较

                iou = Base_Datasets.calculate_iou(current_bbox, bbox_list[j])
                if iou >= iou_threshold:
                    meets_criteria = False
                    break

            if meets_criteria:
                filtered_bboxes.append(current_bbox)
                filtered_id.append(oobjId_list[i])

        return filtered_bboxes, filtered_id

    def get_num(self, set_name):
        '''
            this function provides the number of valid croppings
            only call it after getting the crop_txt files
        '''
        print(f'dataset dir: {self.base_dir}')
        for type in ['pedestrian', 'nonPedestrian']:
            txt_path = os.path.join(self.base_dir, 'crops_txt', set_name, type + '.txt')
            with open(txt_path, 'r') as f:
                data = f.readlines()
            print(f'{set_name} - {type} - {len(data)}')

    def get_crop_txt(self, set_name):
        '''
            set_name: train, val
            this function unifies from source format to target .txt format
                pedestrian.txt: image_path, x1, x2, y1, y2, crop_name
                nonPedestrian.txt: image_path
        '''
        crop_txt_dir = os.path.join(self.base_dir, 'crops_txt', set_name)
        if not os.path.exists(crop_txt_dir):
            os.makedirs(crop_txt_dir)

        print(f'Crop txt saving dir:{crop_txt_dir}.')

        images, labels = self.get_imgAndlabels(set_name)

        print(f'Num of images: {len(images)}, num of labels: {len(labels)}')

        ped_list = []
        nonPed_list = []

        for idx in tqdm(range(len(labels))):
            json_path = labels[idx]
            if 'citypersons' in self.base_dir.lower() and set_name == 'train_extra':
                annos, save_to = self.read_seg_json(json_path, images[idx])
            else:
                annos, save_to = self.get_anno(json_path, images[idx])

            if annos != -1:
                if save_to == 'pedestrian':
                    ped_list.extend(annos)
                else:
                    nonPed_list.extend(annos)

        ped_txt = os.path.join(crop_txt_dir, 'pedestrian.txt')
        with open(ped_txt, 'a') as ped_f:
            for item in ped_list:
                item = item + '\n'
                ped_f.write(item)

        nonPed_txt = os.path.join(crop_txt_dir, 'nonPedestrian.txt')
        with open(nonPed_txt, 'a') as nonPed_f:
            for item in nonPed_list:
                item = item + '\n'
                nonPed_f.write(item)

    def crop_objects(self, set_list, type, crop_num=-1):
        '''
            type: pedestrian, nonPedestrian
            cropping images from txt files
        '''
        cropped_num = 0

        for set_name in set_list:

            crop_save_dir = os.path.join(self.base_dir, 'crops', set_name, type)
            if not os.path.exists(crop_save_dir):
                os.makedirs(crop_save_dir)

            crop_txt_dir = os.path.join(self.base_dir, 'crops_txt', set_name, type + '.txt')
            with open(crop_txt_dir, 'r') as f:
                data = f.readlines()

            random.shuffle(data)

            to_crop_num = len(data) if crop_num == -1 or crop_num > len(data) else crop_num
            to_crop_num = max((to_crop_num - cropped_num), 0)
            if to_crop_num == 0:
                break
            print(f'Cropping {to_crop_num} to {crop_save_dir}')

            for idx in tqdm(range(to_crop_num)):
                cropped_num += 1
                item = data[idx]
                item = item.strip().split(',')
                image_path = item[0]
                image = Image.open(image_path)

                if type == 'pedestrian':
                    temp_list = item[1].split()
                    x1, y1, x2, y2 = map(lambda x: int(float(x)), temp_list)
                    crop_name = item[-1]

                    # to show the croppings
                    # img_draw = ImageDraw.ImageDraw(image)
                    # img_draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=5)
                    # image.show()
                    # break
                else:
                    x1 = random.randint(0, self.image_width - 224)
                    y1 = random.randint(int(self.image_height * 0.2), int(self.image_height * 0.8) - 224)
                    x2 = x1 + 224
                    y2 = y1 + 224
                    image_name = os.path.basename(image_path)
                    crop_name = image_name.split('.')[0] + '.jpg'

                crop_box = (x1, y1, x2, y2)
                crop = image.crop(crop_box)
                crop_save_path = os.path.join(crop_save_dir, crop_name.lstrip())
                crop.save(crop_save_path)

    def split_dataset(self):
        '''
            to split the dataset into train/val/test
            需要提前准备好准备分割的crop数量
        '''

        def write_to_txt(txt_path, data):
            with open(txt_path, 'a') as f:
                for item in data:
                    f.write(item)

        txt_dir = os.path.join(self.base_dir, 'dataset_txt')
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)

        print(f'Txt save dir: {txt_dir}')

        crops_dir = os.path.join(self.base_dir, 'crops')
        set_list = os.listdir(crops_dir)

        # gather img info
        ped_list = []
        nonPed_list = []

        for set_name in set_list:
            set_path = os.path.join(crops_dir, set_name)

            for type_name in os.listdir(set_path):
                image_list = os.listdir(os.path.join(set_path, type_name))
                if type_name == 'pedestrian':
                    ped_list.extend([os.path.join('crops', set_name, type_name, img) + ' 1\n' for img in image_list])
                else:
                    nonPed_list.extend([os.path.join('crops', set_name, type_name, img) + ' 0\n' for img in image_list])

        ped_num = len(ped_list)
        nonPed_num = len(nonPed_list)
        print(f'ped_num:{ped_num}, nonPed:{nonPed_num}')

        random.shuffle(ped_list)
        random.shuffle(nonPed_list)

        # split
        train_set = []
        val_set = []
        test_set = []

        train_set.extend(ped_list[: int(0.6 * ped_num)])
        val_set.extend(ped_list[int(0.6 * ped_num): int(0.8 * ped_num)])
        test_set.extend(ped_list[int(0.8 * ped_num):])

        train_set.extend(nonPed_list[: int(0.6 * nonPed_num)])
        val_set.extend(nonPed_list[int(0.6 * nonPed_num): int(0.8 * nonPed_num)])
        test_set.extend(nonPed_list[int(0.8 * nonPed_num):])

        print(f'Training set num: {len(train_set)}')
        print(f'Validation set num: {len(val_set)}')
        print(f'Test set num: {len(test_set)}')

        write_to_txt(os.path.join(self.base_dir, 'dataset_txt', 'train.txt'), train_set)
        write_to_txt(os.path.join(self.base_dir, 'dataset_txt', 'val.txt'), val_set)
        write_to_txt(os.path.join(self.base_dir, 'dataset_txt', 'test.txt'), test_set)


class Read_ECP(Base_Datasets):
    def __init__(self, base_dir):
        '''
            set_name: train, val
        '''
        super().__init__(base_dir)
        self.base_dir = base_dir

        # CityPersons image size
        self.image_width = 1920
        self.image_height = 1024

        # if these classes occur in an image, the image is not pedestrian-free
        self.ped_cls = ['pedestrian', 'buggy-group', 'person-group-far-away', 'rider', 'rider+vehicle-group-far-away']

    def get_imgAndlabels(self, set_name):
        print(f'Dataset: {self.base_dir}')
        images = []
        labels = []

        for t in ['day', 'night']:
            print(f'Collection from {t} - {set_name}')

            img_path = os.path.join(self.base_dir, t, 'img', set_name)
            label_path = os.path.join(self.base_dir, t, 'labels', set_name)
            city_list = os.listdir(img_path)

            for city in city_list:
                img_dir = os.path.join(img_path, city)
                label_dir = os.path.join(label_path, city)

                images.extend(list(os.path.join(img_dir, cur_img) for cur_img in os.listdir(img_dir)))
                labels.extend(
                    list(os.path.join(label_dir, cur_img.replace('.png', '.json')) for cur_img in os.listdir(img_dir)))

        return images, labels

    def get_anno(self, json_path, image_path):
        '''
            handle single json file
        '''
        anno_list = []
        temp_bbox = []
        temp_objId = []
        ped_flag = False  # whether the image contains pedestrian

        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

            objects = data['children']
            for obj_id, obj in enumerate(objects):
                category = obj['identity']

                if category in self.ped_cls:
                    ped_flag = True

                if category == 'pedestrian' and len(obj['tags']) == 0:
                    x1, y1, x2, y2 = int(obj['x0']), int(obj['y0']), int(obj['x1']), int(obj['y1'])
                    # filtering to small or to big pedestrians
                    w = x2 - x1
                    h = y2 - y1
                    if w > self.ped_min_size and h > self.ped_min_size and w < self.ped_max_size and h < self.ped_max_size:
                        x1, x2 = Base_Datasets.calc_crop(x1, x2, max_val=self.image_width)
                        y1, y2 = Base_Datasets.calc_crop(y1, y2, max_val=self.image_height)
                        if x1 != -1 and y1 != -1:
                            bbox = (x1, y1, x2, y2)
                            temp_bbox.append(bbox)
                            temp_objId.append(obj_id)

            if ped_flag:  # the image contains pedestrian
                bbox_list, filtered_id = Read_BDD100K.filter_bboxes_by_iou(temp_bbox, temp_objId)

                for idx, bbox in enumerate(bbox_list):
                    bbox_msg = ' '.join(map(str, bbox))
                    image_base_name = os.path.basename(image_path).split('.')[0]
                    crop_name = image_base_name + '_' + str(filtered_id[idx]) + '.jpg'
                    # crop_name = image_name.replace('.jpg', '_' + str(filtered_id[idx]) + '.jpg')
                    msg = image_path + ', ' + bbox_msg + ', ' + crop_name
                    anno_list.append(msg)
                return anno_list, 'pedestrian'
            else:  # no pedestrian-related obj in this image
                return [image_path], 'nonPedestrian'


class Read_CityPersons(Base_Datasets):
    def __init__(self, base_dir):
        super().__init__(base_dir)

        self.base_dir = base_dir

        # CityPersons image size
        self.image_width = 2048
        self.image_height = 1024

        # if these classes occur in an image, the image is not pedestrian-free
        self.ped_cls = ['person', 'rider', 'persongroup', 'bicyclegroup', 'ridergroup', 'motorcycle', 'motorcyclegroup']

        self.image_dir_name = 'leftImg8bit_blurred\leftImg8bit_blurred'
        self.label_dir_name = r'gtBboxCityPersons'

    def get_imgAndlabels(self, set_name):
        images = []
        labels = []

        image_dir_path = os.path.join(self.base_dir, self.image_dir_name, set_name)
        label_dir_path = os.path.join(self.base_dir, self.label_dir_name, set_name)

        city_list = os.listdir(image_dir_path)

        for city in city_list:
            img_dir = os.path.join(image_dir_path, city)
            label_dir = os.path.join(label_dir_path, city)

            if set_name == 'train_extra':
                for label_file in os.listdir(label_dir):
                    if label_file.endswith('.json'):
                        labels.append(os.path.join(label_dir, label_file))
                        images.append(os.path.join(img_dir, label_file.replace('_gtCoarse_polygons.json',
                                                                               '_leftImg8bit_blurred.jpg')))
            else:  # for train and val set
                images.extend(list(os.path.join(img_dir, cur_img) for cur_img in os.listdir(img_dir)))
                labels.extend(list(
                    os.path.join(label_dir, cur_img.replace('leftImg8bit_blurred.jpg', 'gtBboxCityPersons.json')) for
                    cur_img in os.listdir(img_dir)))

        return images, labels

    def get_anno(self, json_path, image_path):
        '''
            handle single json file from CityPersons train and val
        '''
        anno_list = []
        temp_bbox = []
        temp_objId = []
        ped_flag = False  # whether the image contains pedestrian

        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

            objects = data['objects']

            for obj in objects:
                category = obj['label']
                if category in self.ped_cls:
                    ped_flag = True
                if category == 'pedestrian':
                    bbox = obj['bboxVis']

                    x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    if w > self.ped_min_size and h > self.ped_min_size and w < self.ped_max_size and h < self.ped_max_size:
                        x2 = x1 + w
                        y2 = y1 + h
                        x1, x2 = Base_Datasets.calc_crop(x1, x2, max_val=self.image_width)
                        y1, y2 = Base_Datasets.calc_crop(y1, y2, max_val=self.image_height)
                        if x1 != -1 and y1 != -1:
                            obj_id = obj['instanceId']
                            bbox = (x1, y1, x2, y2)
                            temp_bbox.append(bbox)
                            temp_objId.append(obj_id)

            if ped_flag:  # the image contains pedestrian
                bbox_list, filtered_id = Read_BDD100K.filter_bboxes_by_iou(temp_bbox, temp_objId)

                for idx, bbox in enumerate(bbox_list):
                    bbox_msg = ' '.join(map(str, bbox))
                    image_name = os.path.basename(image_path)
                    crop_name = image_name.replace('.jpg', '_' + str(filtered_id[idx]) + '.jpg')
                    msg = image_path + ', ' + bbox_msg + ', ' + crop_name
                    anno_list.append(msg)
                return anno_list, 'pedestrian'
            else:  # no pedestrian-related obj in this image
                return [image_path], 'nonPedestrian'

    def read_seg_json(self, json_path, image_path):

        ped_label = ['person', 'persongroup', 'rider', 'bicyclegroup', 'ridergroup', 'motorcycle', 'motorcyclegroup']
        ped_flag = False  # whether the image contains pedestrian
        anno_list = []
        temp_bbox = []
        temp_objId = []

        with open(json_path, 'r') as fcc_file:
            data = json.load(fcc_file)

            objects = data['objects']
            '''
            获取ped
            '''

            for obj_id, obj in enumerate(objects):
                label = obj['label']

                if label in ped_label:
                    ped_flag = True

                polygons = obj['polygon']
                x1, y1 = np.min(polygons, axis=0)  # 按列求最小值
                x2, y2 = np.max(polygons, axis=0)

                label_w = x2 - x1
                label_h = y2 - y1
                w_flag = label_w > self.ped_min_size and label_w < self.ped_max_size
                h_flag = label_h > self.ped_min_size and label_h < self.ped_max_size

                if (label == 'person' or label == 'persongroup') and w_flag and h_flag:
                    x1, x2 = Base_Datasets.calc_crop(x1, x2, max_val=self.image_width)
                    y1, y2 = Base_Datasets.calc_crop(y1, y2, max_val=self.image_height)

                    if x1 != -1 and y1 != -1:
                        bbox = (x1, y1, x2, y2)
                        temp_bbox.append(bbox)
                        temp_objId.append(obj_id)

            if ped_flag:
                bbox_list, filtered_id = Read_BDD100K.filter_bboxes_by_iou(temp_bbox, temp_objId)
                for idx, bbox in enumerate(bbox_list):
                    bbox_msg = ' '.join(map(str, bbox))
                    image_name = os.path.basename(image_path)
                    crop_name = image_name.replace('.jpg', '_' + str(filtered_id[idx]) + '.jpg')
                    msg = image_path + ', ' + bbox_msg + ', ' + crop_name
                    anno_list.append(msg)
                return anno_list, 'pedestrian'
            else:  # no pedestrian-related obj in this image
                return [image_path], 'nonPedestrian'


def decom_BDD100KLabels(json_path):
    '''
        Decomposing the BDD100K .json files (1G takes too long for loading)
    '''
    if 'val' in json_path:
        set_name = 'val'
    elif 'train' in json_path:
        set_name = 'train'
    else:
        raise KeyError(f'Please check json_path:{json_path}')

    print(f'Decomposing {json_path}.')

    label_dir = os.path.dirname(json_path)
    json_save_dir = os.path.join(label_dir, set_name)

    if not os.path.exists(json_save_dir):
        msg = f'Making dir:{json_save_dir}'
        os.mkdir(json_save_dir)
    else:
        msg = f'Saving dir:{json_save_dir}'

    print(msg)
    with open(json_path, 'r', encoding='utf-8') as f_all_json:

        objects = ijson.items(f_all_json, 'item', use_float=True)
        for idx, obj in enumerate(objects):
            name = obj.get('name', '')
            json_name = name.replace('.jpg', '.json')
            json_path = os.path.join(json_save_dir, json_name)

            with open(json_path, 'w') as f_single_json:
                json.dump(obj, f_single_json, ensure_ascii=False, indent=2)

            if (idx + 1) % 1000 == 0:
                print(f"Saving {idx + 1} files")

    print('Finished!')


class Read_BDD100K(Base_Datasets):
    def __init__(self, base_dir):
        super().__init__(base_dir)
        self.base_dir = base_dir
        # self.set_name = set_name

        # BDD100K image size
        self.image_width = 1280
        self.image_height = 720

        # if these classes occur in an image, the image is not pedestrian-free
        self.ped_cls = ['rider', 'person', 'motor', 'bike']

    def get_imgAndlabels(self, set_name):

        img_dir = os.path.join(self.base_dir, 'images', '100k', set_name)
        label_dir = os.path.join(self.base_dir, 'labels', '100k', set_name)

        images = []
        labels = []

        for j_file in os.listdir(label_dir):
            json_path = os.path.join(label_dir, j_file)
            img_path = os.path.join(img_dir, j_file.replace('.json', '.jpg'))

            images.append(img_path)
            labels.append(json_path)

        return images, labels

    def get_anno(self, json_path, image_path):
        '''
            handle single json file
        '''
        anno_list = []
        temp_bbox = []
        temp_objId = []
        ped_flag = False  # whether the image contains pedestrian

        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            objects = data['labels']
            # image_path = os.path.join(self.img_dir, data['name'])
            for obj in objects:
                category = obj['category']
                if category in self.ped_cls:
                    ped_flag = True

                if category == 'person' and not obj['attributes']['occluded']:
                    bbox = obj['box2d']
                    x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
                    # filtering to small or to big pedestrians
                    w = x2 - x1
                    h = y2 - y1
                    if w > self.ped_min_size and h > self.ped_min_size and w < self.ped_max_size and h < self.ped_max_size:
                        x1, x2 = Base_Datasets.calc_crop(x1, x2, max_val=self.image_width)
                        y1, y2 = Base_Datasets.calc_crop(y1, y2, max_val=self.image_height)
                        if x1 != -1 and y1 != -1:
                            obj_id = obj['id']
                            bbox = (x1, y1, x2, y2)
                            temp_bbox.append(bbox)
                            temp_objId.append(obj_id)

            if ped_flag:  # the image contains pedestrian
                bbox_list, filtered_id = Read_BDD100K.filter_bboxes_by_iou(temp_bbox, temp_objId)

                for idx, bbox in enumerate(bbox_list):
                    bbox_msg = ' '.join(map(str, bbox))
                    image_name = os.path.basename(image_path)
                    crop_name = image_name.replace('.jpg', '_' + str(filtered_id[idx]) + '.jpg')
                    msg = image_path + ', ' + bbox_msg + ', ' + crop_name
                    anno_list.append(msg)
                return anno_list, 'pedestrian'
            else:  # no pedestrian-related obj in this image
                return [image_path], 'nonPedestrian'





















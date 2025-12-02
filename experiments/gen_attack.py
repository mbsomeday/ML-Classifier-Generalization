'''
    用attack生成image并保存
'''

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys

curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)


from tqdm import tqdm
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from torchvision.models import efficientnet_b0
import argparse, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
# import matplotlib.pyplot as plt

from data.dataset import my_dataset
from utils.utils import load_model, save_image_tensor, DEVICE


torch.manual_seed(13)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default=r'D:\my_phd\on_git\ML-Classifier-Generalization\aa_test\attack')
    parser.add_argument('--ds_name_list', nargs='+', default=['D1'])
    parser.add_argument('--model_weights', type=str, default=r'D:\my_phd\Model_Weights\Stage6\new_dataset\dsClsD1D2D3-08-1.09839.pth')
    parser.add_argument('--txt_name', type=str, default='train.txt')

    args = parser.parse_args()

    return args


args = get_args()
save_dir = args.save_dir
txt_name = args.txt_name
model_weights = args.model_weights
ds_name_list = args.ds_name_list
batch_size = 1

# 加载模型和数据等
ped_model = efficientnet_b0(num_classes=3, weights=None)
ped_model = load_model(ped_model, model_weights)
ped_model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ped_model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=ped_model,
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 224, 224),
    nb_classes=2,
)

train_dataset = my_dataset(ds_name_list=ds_name_list, path_key='Stage6_org', txt_name=txt_name)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

attack = FastGradientMethod(estimator=classifier, eps=0.07)
plt_transforms = transforms.ToPILImage()

for data_dict in tqdm(train_loader):
    image = data_dict['image']  # tensor [n, 3, 224, 224]
    img_name = data_dict['img_name'][0]
    image_np = image.numpy()        # nparray [1, 3, 224, 224]

    img_label = int(data_dict['ped_label'][0])
    label = 'nonPedestrian' if img_label == 0 else 'pedestrian'

    # print(f'img_name:{img_name}')

    x_test_adv = attack.generate(x=image_np)

    adv_img = x_test_adv[0].transpose((1, 2, 0))
    adv_tensor = torch.from_numpy(adv_img).permute(2, 0, 1).unsqueeze(0)

    # 保存attack图片
    save_path = os.path.join(save_dir, label, img_name)
    save_image_tensor(adv_tensor, save_path)

    # # 结果对比
    # org_out = ped_model(image)
    # print(f'org:{torch.softmax(org_out, dim=1)}')
    # att_out = ped_model(adv_tensor)
    # print(f'att:{torch.softmax(att_out, dim=1)}')
    #
    # # 展示图片
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(plt_transforms(image[0]))
    # plt.title('org')
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.clip(adv_img, 0, 1))
    # plt.title()
    # plt.show()


    # break


# x_test_adv = attack.generate(x=x_test)














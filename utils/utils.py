import importlib, torch
from torchvision import utils as vutils


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_obj_from_str(in_str):
    '''
        loading functions accodring to the input string, e.g. data.build_dataset.Base_Datasets
    '''
    module, cls = in_str.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


class DotDict(dict):
    '''
        将字典转换为可直接用 . 调用的对象
    '''
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

    def __setitem__(self, key, value):
        '''
            保证 dict[key] 的形式可以更改值
        '''
        if isinstance(value, dict):
            value = DotDict(value)
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        '''
            保证 dict.key 的形式可以更改值
        '''
        if isinstance(value, dict):
            value = DotDict(value)
        self[key] = value


def load_model(model, weights_path):
    print(f'Loading model from {weights_path}')
    ckpts = torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
    model.load_state_dict(ckpts['model_state_dict'])
    return model


def save_image_tensor(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    vutils.save_image(input_tensor, filename)






























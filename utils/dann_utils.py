# code borrow from https://github.com/jiaweiww/PyTorch-DANN/blob/main/utils.py

import math, torch
from torch.autograd import Function

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = (grad_output.neg() * ctx.lamda)
        return output, None


def adjust_alpha(i, epoch, min_len, nepochs):
    '''
        sigmoid函数的变体，将alpha逐渐从0增加到1
    '''
    p = float(i + epoch * min_len) / nepochs / min_len
    o = -10
    alpha = 2. / (1. + math.exp(o * p)) - 1
    return alpha


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
























import os


class Base_Trainer():
    '''
        Implementing the general functions for all trainers.
    '''
    def __init__(self, args):
        self.args = args


    def print_args(self):
        '''
            参数打印并保存到txt文件中
        '''
        print('-' * 40 + ' Args ' + '-' * 40)

        info = []
        for k, v in vars(self.args).items():
            msg = f'{k}: {v}'
            print(msg)
            info.append(msg)

        # 将本次实验的参数写入txt中
        write_to_txt = os.path.join(self.callback_save_path, 'Args.txt')
        if os.path.exists(write_to_txt):
            os.remove(write_to_txt)
        with open(write_to_txt, 'a') as f:
            for item in info:
                f.write(item + '\n')

    def decomp_cm(self, cm):
        '''
            对混淆矩阵进行分解
        '''
        tn, fp, fn, tp = cm.ravel()
        return f'{tn}, {fp}, {fn}, {tp}'


















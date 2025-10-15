# 将上级目录加入 sys.path， 防止命令行运行时找不到包
import os, sys
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)


from experiments.pedestrian_classification import Ped_Classifier
from configs.configs import testPedestrian_args


opts = testPedestrian_args()
ds_cls = Ped_Classifier(opts)
ds_cls.test()













from experiments.pedestrian_classification import Ped_Classifier
from configs.configs import testPedestrian_args


opts = testPedestrian_args()
ds_cls = Ped_Classifier(opts)
# ds_cls.test()













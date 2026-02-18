import os, torch

LCA = {
    'org_dataset': {
        'D1': r'/tormenta/s/ssesaai/data/Stage4_D1_ECPDaytime_7Augs',
        'D2': r'/tormenta/s/ssesaai/data/Stage4_D2_CityPersons_7Augs',
        'D3': r'/tormenta/s/ssesaai/data/Stage4_D3_ECPNight_7Augs',
        'D4': r'/tormenta/s/ssesaai/data/Stage4_D4_BDD100K_7Augs'
    },

    'tiny_dataset': {
        'D1': r'/tormenta/s/ssesaai/data/TinyD1',
        'D2': r'/tormenta/s/ssesaai/data/TinyD2',
        'D3': r'/tormenta/s/ssesaai/data/TinyD3',
        'D4': r'/tormenta/s/ssesaai/data/TinyD4',
    },

    'autoencoder_ckpt_dict': {
        'D1': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D1_epo26_00894.ckpt',
        'D2': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D2_epo59_01239.ckpt',
        'D3': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D3_epo49_01236.ckpt',
        'D4': r'/tormenta/s/ssesaai/data/weights/Stage5_LDM/autoencoder/D4_epo34_01236.ckpt'
    }
}

LOCAL = {
    'Stage6_org': {
        'D1': r'D:\my_phd\dataset\Stage6\stage6_ecp',
        'D2': R'D:\my_phd\dataset\Stage6\stage6_citypersons',
        'D3': r'D:\my_phd\dataset\D4_BDD100K\bdd100k'
    },

    'org_dataset': {
        'D1': r'D:\my_phd\dataset\Stage3\D1_ECPDaytime',
        'D2': r'D:\my_phd\dataset\Stage3\D2_CityPersons',
        'D3': r'D:\my_phd\dataset\Stage3\D3_ECPNight',
        'D4': r'D:\my_phd\dataset\Stage3\D4_BDD100K',
    },

    'tiny_dataset': {
        'D1': r'D:\my_phd\dataset\Stage5\TinyD1',
        'D2': r'D:\my_phd\dataset\Stage5\TinyD2',
        'D3': r'D:\my_phd\dataset\Stage5\TinyD3',
        'D4': r'D:\my_phd\dataset\Stage5\TinyD4',
    },

    'ped_cls_ckpt': {
        'D1': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D1-014-0.9740.pth',
        'D2': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D2-025-0.9124.pth',
        'D3': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D3-025-0.9303.pth',
        'D4': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-D4-013-0.9502.pth',
    },

    'EfficientNet_ped_cls': {
        'D1': r'D:\my_phd\Model_Weights\Stage5\EfficientNet_Cls\EfficientB0D1-003-0.982471.pth',
        'D2': r'D:\my_phd\Model_Weights\Stage5\EfficientNet_Cls\EfficientB0D2-011-0.975242.pth',
        'D3': r'D:\my_phd\Model_Weights\Stage5\EfficientNet_Cls\EfficientB0D3-018-0.941520.pth',
        'D4': r'D:\my_phd\Model_Weights\Stage5\EfficientNet_Cls\EfficientB0D4-023-0.963394.pth',
    },

    'EfficientNet_ds_cls': r'D:\my_phd\Model_Weights\Stage5\EfficientNet_Cls\EfficientB0_dsCls-015-0.880432.pth',

    'ds_cls_ckpt': r'D:\my_phd\Model_Weights\Stage4\Baseline\vgg16bn-dsCls-029-0.9777.pth',

    'autoencoder_ckpt_dict': {
        'D1': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D1_epo26_00894.ckpt',
        'D2': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D2_epo59_01239.ckpt',
        'D3': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D3_epo49_01236.ckpt',
        'D4': r'D:\my_phd\Model_Weights\Stage5\LDM_AE_ckpt\D4_epo34_01236.ckpt',
    }
}

KAGGLE = {
    'Stage6_onlyTrain': {
        'D2': r'/kaggle/input/m1-ecp-oppomask',

    },

    'Stage6_org': {
        # 4500
        'D1': r'/kaggle/input/datasets/jiaweiwang802/stage6-dataset-ecp',
        'D2': r'/kaggle/input/datasets/jiaweiwang802/stage6-dataset-citypersons',
        'D3': r'/kaggle/input/datasets/jiaweiwang802/stage6-dataset-bdd100k',
    },

    'Stage6_experiment': {
        'D2': r'/kaggle/input/datasets/jiaweiwang802/stage6-dataset-citypersons-experiment',

    },

    'org_dataset': {
        'D1': r'/kaggle/input/stage4-d1-ecpdaytime-7augs',
        'D2': r'/kaggle/input/stage4-d2-citypersons-7augs',
        'D3': r'/kaggle/input/stage4-d3-ecpnight-7augs',
        'D4': r'/kaggle/input/stage4-d4-7augs'
    },

    'tiny_dataset': {
        # 'D1': r'/kaggle/input/stage5-datasets-tiny/TinyD1',

        'D1': r'/kaggle/input/stage5-tinydsaug/TinyD1Aug',
        'D2': r'/kaggle/input/stage5-tinydsaug/TinyD2Aug',
        'D3': r'/kaggle/input/stage5-tinydsaug/TinyD3Aug',
        'D4': r'/kaggle/input/stage5-tinydsaug/TinyD4Aug',
    },

    'AE_same_dataset': {
        'D1': r'/kaggle/input/stage5-dataset-orgrecons/D1_Recon/D1',
        'D2': r'/kaggle/input/stage5-dataset-orgrecons/D2_Recons/D2',
        'D3': r'/kaggle/input/stage5-dataset-orgrecons/D3_Recons/D3',
        'D4': r'/kaggle/input/stage5-dataset-orgrecons/D4_Recons/D4',
    },

    'AE1_dataset': {
        'D1': r'/kaggle/input/stage5-dataset-orgrecons/D1_Recon/D1',
        'D2': r'/kaggle/input/stage5-dataset-ae1recons/AE1D2_test/D2',
        'D3': r'/kaggle/input/stage5-dataset-ae1recons/AE1D3_test/D3',
        'D4': r'/kaggle/input/stage5-dataset-ae1recons/AE1D4_test/D4',
    },

    'AE4_dataset': {
        'D1': r'/kaggle/input/stage5-dataset-ae4recons/AE4D1_test',
        'D2': r'/kaggle/input/stage5-dataset-ae4recons/AE4D2_test',
        'D3': r'/kaggle/input/stage5-dataset-ae4recons/AE4D3_test',
        'D4': r'/kaggle/input/stage5-dataset-ae4recons/AE4D4_test',
    },

    'EfficientNet_ped_cls': {
        'D1': r'/kaggle/input/stage5-weights-effb0baseline/EfficientB0D1-003-0.982471.pth',
        'D2': r'/kaggle/input/stage5-weights-effb0baseline/EfficientB0D2-011-0.975242.pth',
        'D3': r'/kaggle/input/stage5-weights-effb0baseline/EfficientB0D3-018-0.941520.pth',
        'D4': r'/kaggle/input/stage5-weights-effb0baseline/EfficientB0D4-023-0.963394.pth'
    },

    'EfficientNet_ds_cls': r'/kaggle/input/stage5-weights-effb0baseline/EfficientB0_dsCls-015-0.880432.pth',

    'ds_cls_ckpt': r'/kaggle/input/stage4-dscls-weights/vgg16bn-dsCls-029-0.9777.pth',

    'ped_cls_ckpt': {
        'D1': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D1-014-0.9740.pth',
        'D2': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D2-025-0.9124.pth',
        'D3': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D3-025-0.9303.pth',
        'D4': r'/kaggle/input/stage4-baseline-weights/vgg16bn-D4-013-0.9502.pth'
    },

    'pedCls_efficient_ckpt': {
        'D3': r'/kaggle/working/Stage5_Alpha/ckpt/EfficientB0D3-015-0.950292.pth'
    },

}

NEXUS = {
    'Stage6_org': {
        'D1': r'/data/jcampos/jiawei_data/datasets/stage6_ecp',
        'D2': r'/data/jcampos/jiawei_data/datasets/stage6_citypersons',
        'D3': r'/data/jcampos/jiawei_data/datasets/stage6BDD100K',
    },

    'org_dataset': {
        'D1': r'/data/jcampos/jiawei_data/datasets/D1',
        'D2': r'/data/jcampos/jiawei_data/datasets/D2',
        'D3': r'/data/jcampos/jiawei_data/datasets/D3',
        'D4': r'/data/jcampos/jiawei_data/datasets/D4'
    },

    'ds_cls_ckpt': r'/data/jcampos/jiawei_data/model_weights/Stage4/vgg16bn-dsCls-029-0.9777.pth'

}

deucalion = {
    'Stage6_org': {
        'D1': r'/projects/F202407660IACDCF2/jiawei_deu_data/datasets/Stage6_ECP',
        'D2': r'/projects/F202407660IACDCF2/jiawei_deu_data/datasets/Stage6_CityPersons',
        'D3': r'/projects/F202407660IACDCF2/jiawei_deu_data/datasets/Stage6_BDD100K',
    },
}

CISUC_Cluster = {
    'Stage6_org': {
        'D2': r'd2',
    },
}



cwd = os.getcwd()

print('-' * 50)

if 'my_phd' in cwd:
    print(f'Run on Local -- working dir: {cwd}')
    PATHS = LOCAL
elif 'kaggle' in cwd:
    print(f'Run on kaggle -- working dir: {cwd}')
    PATHS = KAGGLE
elif 'veracruz' in cwd:
    print(f'Run on lca -- working dir: {cwd}')
    PATHS = LCA
elif 'jiawei_data' in cwd:
    print(f'Run on Nexus Server -- working dir: {cwd}')
    PATHS = NEXUS
elif 'F202407660IACDCF2' in cwd:
    print(f'Run on deucalion -- working dir: {cwd}')
    PATHS = deucalion
else:
    # PATHS = CISUC_Cluster
    raise Exception('运行平台未知，需配置路径!')
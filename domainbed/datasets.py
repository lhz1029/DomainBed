# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from os.path import dirname, abspath
import os
import torch
from easydict import EasyDict as edict
import json
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate
from .utils import transform, GetTransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    'chestXR',
    'JointDataset'
]

all_diseases = ['Atelectasis', 'Cardiomegaly',
                'Consolidation', 'Edema', 'Pneumonia']
diseases = ['Pneumonia']
csv_path = '/scratch/wz727/chestXR/data/labels/'

domainbed_path = dirname(dirname(abspath(__file__)))
print('In folder = {}'.format(domainbed_path))


class MultipleDomainDataset:
    N_STEPS = 5001  # Default, subclasses may override
    CHECKPOINT_FREQ = 100  # Default, subclasses may override
    N_WORKERS = 8  # Default, subclasses may override
    ENVIRONMENTS = None  # Subclasses should override
    INPUT_SHAPE = None  # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class ChestDataset(Dataset):
    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        # print(self._image_paths[idx])
        image = Image.fromarray(image)
        # image = transform(image.transpose((2, 0, 1)), self.cfg)
        # try:
        #     image = Image.fromarray(image)
        # except:
        #     raise Exception('None image path: {}'.format(self._image_paths[idx]))
        resize = transforms.Resize((256, 256))
        to_tensor = transforms.ToTensor()
        image = to_tensor(resize(image))
        if self._mode == 'train':
            image = GetTransforms(image, type="Aug")
        # print(idx, type(idx))
        labels = np.array([self._labels[idx]]).astype(np.float32)
        path = self._image_paths[idx]
        if self._mode == 'train' or self._mode == 'val' or self._mode == 'test':
            return (image, labels)
        elif self._mode == 'output':
            return (image, path)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))

    def upsample(self):
        # if self._mode == 'train' and upsample:
        ratio = (len(self._labels) - self._labels.sum(axis=0)
                 ) // self._labels.sum(axis=0) - 1
        ratio = ratio[self._idx][0]
        pos_idx = np.where(self._labels[:, self._idx] == 1)[0]
        if ratio >= 1:
            up_idx = np.concatenate(
                (np.arange(len(self._labels)), np.repeat(pos_idx, ratio)))
            self._image_paths = self._image_paths[up_idx]
            self._labels = self._labels[up_idx]


class JointDataset(MultipleDomainDataset):
    # def __init__(self, data_path=None, mode='train', upsample=True, subset=True):
    #     label_paths={
    #         'mimic':"/scratch/wz727/chestXR/data/mimic-cxr/train_sub.csv",
    #         'chexpert":"/CheXpert-v1.0/train.csv"
    #     }
    #     datasets = {}
    #     cfg = 'configs/chexpert_config.json'
    #     with open(cfg) as f:
    #         self.cfg = edict(json.load(f))
    #     for key, path in label_paths:
    #         if key=='mimic':
    #             datasets['mimic'] = MimicCXRDataset(path)
    #         elif key=='chexpert'
    #             datasets['chexpert'] = CheXpertDataset(path)
    #         else:
    #             assert False, 'Unrecognized dataset'
    #     self._mode = mode
    #     for key in datasets.keys():
    #         self._labels.extend(datasets[key]._labels)
    #         self._image_paths.extend(datasets[key]._image_paths)
    # class chestXR(MultipleDomainDataset):
    ENVIRONMENTS = ['mimic-cxr', 'chexpert']
    N_STEPS = 100000  # Default, subclasses may override
    CHECKPOINT_FREQ = 5000  # Default, subclasses may override
    N_WORKERS = 8

    def __init__(self, root, test_envs, mode, hparams):
        super().__init__()
        #     label_paths={
        #         'mimic':"/scratch/wz727/chestXR/data/mimic-cxr/train_sub.csv",
        #         'chexpert":"/CheXpert-v1.0/train.csv"
        #     }
        # paths = ['/beegfs/wz727/mimic-cxr',
        #          '/scratch/wz727/chest_XR/chest_XR/data/CheXpert',
        #          '/scratch/wz727/chest_XR/chest_XR/data/chestxray8',
        #          '/scratch/lhz209/padchest']
        # paths = ['/scratch/wz727/chestXR/data/mimic-cxr', '', '/chestxray8', '/padchest']
        print('CALLING MIMIC')
        mimic = MimicCXRDataset("/scratch/wz727/chestXR/data/mimic-cxr/train_sub.csv",
                                mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])

        print('CALLING CHEXPERT')
        chexpert = CheXpertDataset('/CheXpert-v1.0/train_sub.csv', mode=mode,
                                   upsample=hparams['upsample'], subset=hparams['subset'])

        # TODO: setup ability to call other datasets

        chexpert_x = chexpert._image_paths
        chexpert_y = chexpert._labels.ravel().astype(np.int)

        mimic_x = mimic._image_paths
        mimic_y = mimic._labels.ravel().astype(np.int)

        # get counts
        print('GETTING COUNTS')
        mimic_count = np.sum(mimic_y)
        chexpert_count = np.sum(chexpert_y)
        print('Datasets original Y means {}, {}'.format(
            mimic_y.mean(), chexpert_y.mean()))
        mimic_healthy_count = mimic_y.shape[0] - mimic_count
        chexpert_healthy_count = chexpert_y.shape[0] - chexpert_count

        # find disease case indices
        mimic_disease_indices = np.arange(mimic_y.shape[0])[mimic_y > 0]
        chexpert_disease_indices = np.arange(
            chexpert_y.shape[0])[chexpert_y > 0]

        # find healthy case indices
        mimic_healthy_indices = np.arange(mimic_y.shape[0])[mimic_y == 0]
        chexpert_healthy_indices = np.arange(
            chexpert_y.shape[0])[chexpert_y == 0]
        
        def _sample_with_prevalence_diffs(indices_a, indices_b, hold=0.9, rho=0.9):
            indices_a_keep_count = int(hold*len(indices_a))
            total_count = int(indices_a_keep_count/rho)
            indices_b_keep_count = total_count - indices_a_keep_count
            if indices_b_keep_count >= len(indices_b):
                _b, _a = _sample_with_prevalence_diffs(indices_b, indices_a, hold, 1-rho)
                print('FLIPPING')
                return _a, _b

            return np.random.choice(indices_a, indices_a_keep_count), np.random.choice(indices_b, indices_b_keep_count)

        # CONSTRUCT THE TRAIN DISTRIBUTION
        print('SET TRAIN PREVALENCE')
        mimic_healthy_train, chexpert_healthy_train = _sample_with_prevalence_diffs(
            mimic_healthy_indices, chexpert_healthy_indices, rho=0.1)

        mimic_disease_train, chexpert_disease_train = _sample_with_prevalence_diffs(
            mimic_disease_indices, chexpert_disease_indices, rho=0.9)

        mimic_train = np.concatenate((mimic_healthy_train, mimic_disease_train))
        chexpert_train = np.concatenate((chexpert_healthy_train, chexpert_disease_train))

        # GET SAMPLES NOT IN THE TRAIN DISTRIBUTION
        mimic_healthy_remain = np.setdiff1d(mimic_healthy_indices, mimic_healthy_train)
        chexpert_healthy_remain = np.setdiff1d(chexpert_healthy_indices, chexpert_healthy_train)

        mimic_disease_remain = np.setdiff1d(mimic_disease_indices, mimic_disease_train)
        chexpert_disease_remain = np.setdiff1d(chexpert_disease_indices, chexpert_disease_train)

        # CONSTRUCT THE TEST DISTRIBUTION
        print('SET TEST PREVALENCE')
        mimic_healthy_test, chexpert_healthy_test = _sample_with_prevalence_diffs(
            mimic_healthy_remain, chexpert_healthy_remain, rho=0.9)

        mimic_disease_test, chexpert_disease_test = _sample_with_prevalence_diffs(
            mimic_disease_remain, chexpert_disease_remain, rho=0.1)

        mimic_test = np.concatenate((mimic_healthy_test, mimic_disease_test))
        chexpert_test = np.concatenate((chexpert_healthy_test, chexpert_disease_test))

        # config creation
        chexpert_cfgstr = domainbed_path + '/configs/chexpert_config.json'
        with open(chexpert_cfgstr) as f:
            chexpert_cfg = edict(json.load(f))

        mimic_cfgstr = domainbed_path + '/configs/mimic_config.json'
        with open(mimic_cfgstr) as f:
            mimic_cfg = edict(json.load(f))

        cxr_train = ChestDataset()
        cxr_train._image_paths = np.concatenate( (mimic_x[mimic_train], chexpert_x[chexpert_train]))
        cxr_train._labels = np.concatenate( (mimic_y[mimic_train], chexpert_y[chexpert_train])).reshape(-1, 1)
        cxr_train._idx = np.array([diseases.index(d) for d in diseases])
        cxr_train.upsample()
        cxr_train.cfg = mimic_cfg
        cxr_train._num_image = len(cxr_train._labels)
        cxr_train._mode = mode

        cxr_test = ChestDataset()
        cxr_test._image_paths = np.concatenate( (mimic_x[mimic_test], chexpert_x[chexpert_test]))
        cxr_test._labels = np.concatenate( (mimic_y[mimic_test], chexpert_y[chexpert_test])).reshape(-1, 1) 
        cxr_test._idx = np.array([diseases.index(d) for d in diseases])
        cxr_test.upsample()
        cxr_test.cfg = mimic_cfg
        cxr_test._num_image = len(cxr_test._labels)
        cxr_test._mode = mode

        self.datasets = [cxr_train, cxr_test]
        print('Datasets generated with Y means {}, {}'.format( cxr_train._labels.mean(), cxr_test._labels.mean()))
        print('Datasets generated with Y shapes {}, {}'.format( cxr_train._labels.shape, cxr_test._labels.shape))

        self.input_shape = (1, 256, 256,)
        self.num_classes = 1


class CheXpertDataset(ChestDataset):
    def __init__(self, label_path, cfg=domainbed_path + '/configs/chexpert_config.json', mode='train', upsample=True, subset=True):
        def get_labels(labels):
            all_labels = []
            for _, row in labels.iterrows():
                all_labels.append([row[d] in [1, -1] for d in diseases])
            return all_labels

        def get_image_paths(labels):
            self._data_path = label_path.rsplit('/', 2)[0]
            all_paths = []
            for _, row in labels.iterrows():
                all_paths.append(self._data_path + '/' + row['Path'])
            return all_paths

        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        labels_path = csv_path + 'chexpert_{}.csv'.format(mode)
        df = pd.read_csv(labels_path)
        # subsetting the data
        # TODO: validation at some point should not be subsetted
        if subset:
            uncertain_diseases = [
                d for d in diseases if d in ['Atelectasis', 'Edema']]
            if uncertain_diseases:
                mask = (df[diseases + ['No Finding']] ==
                        1).any(1) | (df[uncertain_diseases] == -1).any(1)
            else:
                mask = (df[diseases + ['No Finding']] == 1).any(1)
            labels = df[mask]
        else:
            labels = df
        labels.fillna(0, inplace=True)
        self._labels = get_labels(labels)
        self._image_paths = get_image_paths(labels)
        self._idx = np.array([diseases.index(d) for d in diseases])
        self._image_paths = np.array(self._image_paths)
        self._labels = np.array(self._labels)
        if self._mode == 'train' and upsample:
            ratio = (len(self._labels) - self._labels.sum(axis=0)
                     ) // self._labels.sum(axis=0) - 1
            ratio = ratio[self._idx][0]
            print('IDX THING HAPPENING')
            pos_idx = np.where(self._labels[:, self._idx] == 1)[0]
            if ratio >= 1:
                up_idx = np.concatenate(
                    (np.arange(len(self._labels)), np.repeat(pos_idx, ratio)))
                self._image_paths = self._image_paths[up_idx]
                self._labels = self._labels[up_idx]
        self._labels = self._labels[:, self._idx]
        self._num_image = len(self._image_paths)
        print('CONSTRUCTED CHEXPERT DATA WITH LABEL MEAN {}/ SHAPE {}'.format(
            self._labels.ravel().mean(), self._labels.shape))


class MimicCXRDataset(ChestDataset):
    def __init__(self, label_path, cfg=domainbed_path+'/configs/mimic_config.json', mode='train', upsample=True, subset=True):
        def get_labels(labels):
            all_labels = []
            for _, row in labels.iterrows():
                all_labels.append([row[d] in [1, -1] for d in diseases])
            return all_labels

        def get_image_paths(labels):
            all_paths = []
            for _, row in labels.iterrows():
                if int(str(row['subject_id'])[:2]) < 15:
                    data_path = '/mimic-cxr_1'
                else:
                    data_path = '/mimic-cxr_2'
                all_paths.append(
                    data_path + '/p' + str(row['subject_id'])[:2] + '/p' + str(row['subject_id']) +
                    '/s' + str(row['study_id']) + '/' +
                    str(row['dicom_id']) + '.jpg'
                )
            return all_paths

        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        label_path = csv_path + 'mimic_{}.csv'.format(mode)
        df = pd.read_csv(label_path)
        # subsetting the data
        # TODO: validation at some point should not be subsetted
        if subset:
            uncertain_diseases = [
                d for d in diseases if d in ['Atelectasis', 'Edema']]
            if uncertain_diseases:
                mask = (df[diseases + ['No Finding']] ==
                        1).any(1) | (df[uncertain_diseases] == -1).any(1)
            else:
                mask = (df[diseases + ['No Finding']] == 1).any(1)
            labels = df[mask]
        else:
            labels = df
        labels = labels[pd.notnull(labels['dicom_id'])]
        labels.fillna(0, inplace=True)
        self._labels = get_labels(labels)
        self._image_paths = get_image_paths(labels)
        self._idx = np.array([diseases.index(d) for d in diseases])
        self._image_paths = np.array(self._image_paths)
        self._labels = np.array(self._labels)
        if self._mode == 'train' and upsample:
            ratio = (len(self._labels) - self._labels.sum(axis=0)
                     ) // self._labels.sum(axis=0) - 1
            ratio = ratio[self._idx][0]
            pos_idx = np.where(self._labels[:, self._idx] == 1)[0]
            if ratio >= 1:
                up_idx = np.concatenate(
                    (np.arange(len(self._labels)), np.repeat(pos_idx, ratio)))
                self._image_paths = self._image_paths[up_idx]
                self._labels = self._labels[up_idx]
        # self._labels = self._labels[:, self._idx]
        self._num_image = len(self._image_paths)
        print('CONSTRUCTED MIMIC DATA WITH LABEL MEAN {}/ SHAPE {}'.format(
            self._labels.ravel().mean(), self._labels.shape))


class ChestXR8Dataset(ChestDataset):
    def __init__(self, label_path, cfg='configs/chestxray8_config.json', mode='train', upsample=True, subset=True):
        def get_labels(label_strs):
            all_labels = []
            for label in label_strs:
                labels_split = label.split('|')
                label_final = [d in labels_split for d in diseases]
                all_labels.append(label_final)
            return all_labels

        self._data_path = label_path.rsplit('/', 1)[0]
        self._mode = mode
        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        label_path = csv_path + 'chestxray8_{}.csv'.format(mode)
        labels = pd.read_csv(label_path)
        if subset:
            labels = labels[labels['Finding Labels'].str.contains(
                '|'.join(diseases + ['No Finding']))]
        if self._mode == 'train' and upsample:
            # labels_neg = labels[labels['Finding Labels'].str.contains('No Finding')]
            # labels_pos = labels[~labels['Finding Labels'].str.contains('No Finding')]
            # one vs all
            labels_pos = labels[labels['Finding Labels'].str.contains(
                diseases[0])]
            labels_neg = labels[~labels['Finding Labels'].str.contains(
                diseases[0])]
            upweight_ratio = len(labels_neg)//len(labels_pos)
            if upweight_ratio > 0:
                labels_pos = labels_pos.loc[labels_pos.index.repeat(
                    upweight_ratio)]
                labels = pd.concat([labels_neg, labels_pos])
        self._image_paths = [os.path.join(
            self._data_path, 'images', name) for name in labels['Image Index'].values]
        self._labels = get_labels(labels['Finding Labels'].values)
        self._num_image = len(self._image_paths)
        # assert len(
        # self._image_paths) == self._num_image, f"Paths and labels misaligned: {(len(self._image_paths), self._num_image)}"


class PadChestDataset(ChestDataset):
    def __init__(self, label_path, cfg='configs/padchest_config.json', mode='train', upsample=True, subset=True):
        def get_labels(label_strs):
            all_labels = []
            for label in label_strs:
                label_final = [d.lower() in label for d in diseases]
                all_labels.append(label_final)
            return all_labels

        self._data_path = label_path.rsplit('/', 1)[0]
        self._mode = mode
        with open(cfg) as f:
            self.cfg = edict(json.load(f))
        label_path = csv_path + 'padchest_{}.csv'.format(mode)
        labels = pd.read_csv(label_path)
        positions = ['AP', 'PA', 'ANTEROPOSTERIOR', 'POSTEROANTERIOR']
        labels = labels[
            pd.notnull(labels['ViewPosition_DICOM']) & labels['ViewPosition_DICOM'].str.match('|'.join(positions))]
        labels = labels[pd.notnull(labels['Labels'])]
        if subset:
            labels = labels[labels['Labels'].str.contains(
                '|'.join([d.lower() for d in diseases] + ['normal']))]

        if self._mode == 'train' and upsample:
            # labels_neg = labels[labels['Labels'].str.contains('normal')]
            # labels_pos = labels[~labels['Labels'].str.contains('normal')]
            # one vs all
            labels_pos = labels[labels['Labels'].str.contains(
                diseases[0].lower())]
            labels_neg = labels[~labels['Labels'].str.contains(
                diseases[0].lower())]
            upweight_ratio = len(labels_neg)//len(labels_pos)
            if upweight_ratio > 0:
                labels_pos = labels_pos.loc[labels_pos.index.repeat(
                    upweight_ratio)]
                labels = pd.concat([labels_neg, labels_pos])
        self._image_paths = [os.path.join(
            self._data_path, name) for name in labels['ImageID'].values]
        self._labels = get_labels(labels['Labels'].values)
        self._num_image = len(self._image_paths)
        # assert len(
        # self._image_paths) == self._num_image, f"Paths and labels misaligned: {(len(self._image_paths), self._num_image)}"


class chestXR(MultipleDomainDataset):
    ENVIRONMENTS = ['mimic-cxr', 'chexpert', 'chestxr8', 'padchest']
    N_STEPS = 100000  # Default, subclasses may override
    CHECKPOINT_FREQ = 5000  # Default, subclasses may override
    N_WORKERS = 8

    def __init__(self, root, test_envs, mode, hparams):
        super().__init__()
        # paths = ['/beegfs/wz727/mimic-cxr',
        #          '/scratch/wz727/chest_XR/chest_XR/data/CheXpert',
        #          '/scratch/wz727/chest_XR/chest_XR/data/chestxray8',
        #          '/scratch/lhz209/padchest']
        paths = ['/scratch/wz727/chestXR/data/mimic-cxr',
                 '', '/chestxray8', '/padchest']
        self.datasets = []
        for i, environment in enumerate(chestXR.ENVIRONMENTS):
            print(environment)
            path = os.path.join(root, environment)
            if environment == 'mimic-cxr':
                env_dataset = MimicCXRDataset(
                    paths[i] + '/train_sub.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
            elif environment == 'chexpert':
                env_dataset = CheXpertDataset(
                    paths[i] + '/CheXpert-v1.0/train_sub.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
            elif environment == 'chestxr8':
                env_dataset = ChestXR8Dataset(
                    paths[i] + '/Data_Entry_2017_v2020.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
            elif environment == 'padchest':
                env_dataset = PadChestDataset(
                    paths[i] + '/padchest_labels.csv', mode=mode, upsample=hparams['upsample'], subset=hparams['subset'])
            else:
                raise Exception('Unknown environments')
            if mode != 'train':
                env_dataset.cfg.use_transforms_type = 'None'

            self.datasets.append(env_dataset)

        self.input_shape = (3, 512, 512,)
        self.num_classes = 1


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )


class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']


# class Debug224(Debug):
#     INPUT_SHAPE = (3, 224, 224)
#     ENVIRONMENTS = ['0', '1', '2']
#
#
# class MultipleEnvironmentMNIST(MultipleDomainDataset):
#     def __init__(self, root, environments, dataset_transform, input_shape,
#                  num_classes):
#         super().__init__()
#         if root is None:
#             raise ValueError('Data directory not specified!')
#
#         original_dataset_tr = MNIST(root, train=True, download=True)
#         original_dataset_te = MNIST(root, train=False, download=True)
#
#         original_images = torch.cat((original_dataset_tr.data,
#                                      original_dataset_te.data))
#
#         original_labels = torch.cat((original_dataset_tr.targets,
#                                      original_dataset_te.targets))
#
#         shuffle = torch.randperm(len(original_images))
#
#         original_images = original_images[shuffle]
#         original_labels = original_labels[shuffle]
#
#         self.datasets = []
#
#         for i in range(len(environments)):
#             images = original_images[i::len(environments)]
#             labels = original_labels[i::len(environments)]
#             self.datasets.append(dataset_transform(images, labels, environments[i]))
#
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#
#
# class ColoredMNIST(MultipleEnvironmentMNIST):
#     ENVIRONMENTS = ['+90%', '+80%', '-90%']
#
#     def __init__(self, root, test_envs, hparams):
#         super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
#                                            self.color_dataset, (2, 28, 28,), 2)
#
#         self.input_shape = (2, 28, 28,)
#         self.num_classes = 2
#
#     def color_dataset(self, images, labels, environment):
#         # # Subsample 2x for computational convenience
#         # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
#         # Assign a binary label based on the digit
#         labels = (labels < 5).float()
#         # Flip label with probability 0.25
#         labels = self.torch_xor_(labels,
#                                  self.torch_bernoulli_(0.25, len(labels)))
#
#         # Assign a color based on the label; flip the color with probability e
#         colors = self.torch_xor_(labels,
#                                  self.torch_bernoulli_(environment,
#                                                        len(labels)))
#         images = torch.stack([images, images], dim=1)
#         # Apply the color to the image by zeroing out the other color channel
#         images[torch.tensor(range(len(images))), (
#                                                          1 - colors).long(), :, :] *= 0
#
#         x = images.float().div_(255.0)
#         y = labels.view(-1).long()
#
#         return TensorDataset(x, y)
#
#     def torch_bernoulli_(self, p, size):
#         return (torch.rand(size) < p).float()
#
#     def torch_xor_(self, a, b):
#         return (a - b).abs()
#
#
# class RotatedMNIST(MultipleEnvironmentMNIST):
#     ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
#
#     def __init__(self, root, test_envs, hparams):
#         super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
#                                            self.rotate_dataset, (1, 28, 28,), 10)
#
#     def rotate_dataset(self, images, labels, angle):
#         rotation = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
#                                                resample=Image.BICUBIC)),
#             transforms.ToTensor()])
#
#         x = torch.zeros(len(images), 1, 28, 28)
#         for i in range(len(images)):
#             x[i] = rotation(images[i])
#
#         y = labels.view(-1)
#
#         return TensorDataset(x, y)
#
#
# class MultipleEnvironmentImageFolder(MultipleDomainDataset):
#     def __init__(self, root, test_envs, augment, hparams):
#         super().__init__()
#         environments = [f.name for f in os.scandir(root) if f.is_dir()]
#         environments = sorted(environments)
#
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#
#         augment_transform = transforms.Compose([
#             # transforms.Resize((224,224)),
#             transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
#             transforms.RandomGrayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#         self.datasets = []
#         for i, environment in enumerate(environments):
#
#             if augment and (i not in test_envs):
#                 env_transform = augment_transform
#             else:
#                 env_transform = transform
#
#             path = os.path.join(root, environment)
#             env_dataset = ImageFolder(path,
#                                       transform=env_transform)
#
#             self.datasets.append(env_dataset)
#
#         self.input_shape = (3, 224, 224,)
#         self.num_classes = len(self.datasets[-1].classes)
#
#
# class VLCS(MultipleEnvironmentImageFolder):
#     CHECKPOINT_FREQ = 300
#     ENVIRONMENTS = ["C", "L", "S", "V"]
#
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "VLCS/")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
#
#
# class PACS(MultipleEnvironmentImageFolder):
#     CHECKPOINT_FREQ = 300
#     ENVIRONMENTS = ["A", "C", "P", "S"]
#
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "PACS/")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
#
#
# class DomainNet(MultipleEnvironmentImageFolder):
#     CHECKPOINT_FREQ = 1000
#     ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
#
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "domain_net/")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
#
#
# class OfficeHome(MultipleEnvironmentImageFolder):
#     CHECKPOINT_FREQ = 300
#     ENVIRONMENTS = ["A", "C", "P", "R"]
#
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "office_home/")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
#
#
# class TerraIncognita(MultipleEnvironmentImageFolder):
#     CHECKPOINT_FREQ = 300
#     ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
#
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "terra_incognita/")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)
#
#
# class SVIRO(MultipleEnvironmentImageFolder):
#     CHECKPOINT_FREQ = 300
#     ENVIRONMENT_NAMES = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
#
#     def __init__(self, root, test_envs, hparams):
#         self.dir = os.path.join(root, "sviro/")
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


# class chestXR(MultipleEnvironmentImageFolder):
#     CHECKPOINT_FREQ = 1000
#     ENVIRONMENTS = ['mimic-cxr', 'chexpert', 'chestxr8', 'padchest']
#     def __init__(self, root, test_envs, hparams):
#         self.dir = ['/beegfs/wz727/mimic-cxr',
#                     '/scratch/wz727/chest_XR/chest_XR/data/CheXpert',
#                     '/scratch/wz727/chest_XR/chest_XR/data/chestxray8',
#                     '/scratch/wz727/chest_XR/chest_XR/data/PadChest']
#         super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import h5py
from pathlib import Path
import os
import torch
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from conformer import FCUUp
from timm.models import create_model
from datasets import build_dataset
from samplers import RASampler
import utils
import models

# from fvcore.nn import FlopCountAnalysis

import cv2
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob




class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root="./datasets/train2014"):
        self.files = glob.glob(root+'/*jpg')
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),])

        #print(self.files, root+'/*jpg')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_file = self.files[index]
        img = cv2.imread(img_file)
        img = Image.fromarray(img)

        if self.transform != None:
            img = self.transform(img)
        return img, 0, img_file.split("/")[-1]

class clevr_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path="change_caption_data/clevr"):
        self.data_path = data_path
        self.file_list = glob.glob(data_path+"/**/*.png")
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_file = self.file_list[index]
        img = cv2.imread(img_file)
        img = Image.fromarray(img)
        img = self.transform(img)
        if 'default' in img_file:
            label = 0
        elif 'nonsemantic' in img_file:
            label = 0
        else: #semantic
            label = 1
        print(img_file)
        return label, img, img_file.split("/")[-1]

class clevr_Dataset_v2(torch.utils.data.Dataset):
    def __init__(self, data_path="change_caption_data/clevr", mode='train'):
        with open("change_caption_data/clevr/splits.json", "r") as json_file:
            self.json_file = json.load(json_file)

        self.mode = mode
        self.data_path = data_path
        self.default_list = glob.glob(data_path+"/images/*.png")
        self.nonsemantic_list = glob.glob(data_path+"/nsc_images/*.png")
        self.semantic_list = glob.glob(data_path + "/sc_images/*.png")

        self.train_list = self.get_images_list(key_list=self.json_file['train'])
        self.val_list = self.get_images_list(key_list=self.json_file['val'])
        self.test_list = self.get_images_list(key_list=self.json_file['test'])
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),])
        if self.mode == 'train':
            self.file_list = self.train_list
        elif self.mode == 'test':
            self.file_list = self.test_list
        else:
            self.file_list = self.val_list

    def get_images_list(self, key_list):
        data_list = []
        for key in key_list:
            str_key = self.convert_key_into_str(key=key)
            data_list.append(self.data_path + "/images/" + "CLEVR_default_" + str_key + ".png")
            data_list.append(self.data_path + "/nsc_images/" + "CLEVR_nonsemantic_" + str_key + ".png")
            data_list.append(self.data_path + "/sc_images/" + "CLEVR_semantic_" + str_key + ".png")
        return data_list

    def convert_key_into_str(self, key):
        str_key = str(key)
        while len(str_key) < 6:
            str_key = '0' + str_key
        return str_key

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_file = self.file_list[index]
        img = cv2.imread(img_file)
        img = Image.fromarray(img)
        img = self.transform(img)
        if 'default' in img_file:
            label = 0
        elif 'nonsemantic' in img_file:
            label = 0
        else:  # semantic
            label = 1
        return img, label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file="datasets/features/CONFORMER_COCO2014_t.hdf5", coco="datasets/train2014/"):
        self.f = h5py.File("datasets/features/CONFORMER_COCO2014.hdf5")
        self.f_t = h5py.File("datasets/features/CONFORMER_COCO2014_t.hdf5")

        self.conformer = create_model("Conformer_base_patch16",
                                  pretrained=False,
                                  num_classes=1000,
                                  drop_rate=0,
                                  drop_path_rate=0.1,
                                  drop_block_rate=None,)
        self.conformer.load_state_dict(torch.load("weights/Conformer_base_patch16.pth"))
        print("load done")

        self.feat_list = {'im': [], 'feat': []}
        self.feat_list_t = {'im': [], 'feat': []}
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),])
        for key in self.f_t:
            if key[-4:] == 'feat':
                self.feat_list_t['feat'].append(key)
                self.feat_list_t['im'].append(coco + key.split('_feat')[0])

        for key in self.f:
            if key[-4:] == 'feat':
                self.feat_list['feat'].append(key)
                self.feat_list['im'].append(coco + key.split('_feat')[0])

    def __len__(self):
        return len(self.feat_list_t['feat'])

    def __getitem__(self, index):
        key = self.feat_list['feat'][index]
        key2 = self.feat_list_t['feat'][index]
        im = self.feat_list_t['im'][index]
        im = cv2.imread(im)
        im = Image.fromarray(im)
        im = self.transform(im)
        feat1 = np.array(self.f[key])
        feat1_t = np.array(self.f_t[key2])

        feat2 = self.conformer.conv_trans_12.expand_block(torch.from_numpy(feat1_t), 14, 14)
        feat3 = self.conformer.conv_trans_12.fusion_block(torch.from_numpy(feat1).float(), feat2, return_x_2=False)
        return feat1, feat2, feat3, im

class CocoCaption(torch.utils.data.Dataset):
    def __init__(self, tokenizer, mode='train'):
        self.tokenizer = tokenizer
        self.mode = mode
        if mode == 'train':
            self.im_path = "./datasets/coco/train2017"
            self.annotations_path = "./datasets/coco/annotations/captions_train2017.json"
        else:
            self.im_path = "./datasets/coco/val2017"
            self.annotations_path = "./datasets/coco/annotations/captions_val2017.json"
        self.files = glob.glob(self.im_path+'/*jpg')
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4452, 0.4457, 0.4464), (0.2592, 0.2596, 0.2600)),])
        with open(self.annotations_path, "r") as json_anot:
            self.annotations = json.load(json_anot)

        #print(self.files, root+'/*jpg')

    def __len__(self):
        return len(self.annotations['annotations'])

    def __getitem__(self, index):
        annotation = self.annotations['annotations'][index]
        img_id = annotation['image_id']
        img_id = str(img_id)
        while len(img_id) < 12:
            img_id = '0' + img_id
        img_name = img_id + '.jpg'
        img_file = os.path.join(self.im_path, img_name)
        img = cv2.imread(img_file)
        img = Image.fromarray(img)

        if self.transform != None:
            img = self.transform(img)
        return img, '<|endoftext|>' + annotation['caption'] + '<|endoftext|>'



# for anot_info in anot_infos:
#     if anot_info['image_id'] == 203564:
#         print(anot_info)
# for img_info in img_infos:
#     if img_info['id'] == 203564:
#         print(img_info)


# print(annotations['images'][0])
# print(annotations['annotations'][0])
# print(len(annotations['annotations']))
# data = Dataset()
# print(data[0][0].shape)
# print(data[0][1].shape)
# print(data[0][2].shape)
# print(data[0][3].shape)

# list = list(ckp.keys())
# for val in list:
#     if "12.expand_block" in val:
#         print(val, ": ", ckp[val].shape)
#
# print(dataset[0][0].shape)
from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import json
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image


class MyDBDataset:
    def __init__(self, root_pth = "/home/test/workspace/experiment/dataset/ocelot2023_v0.1.1/", cell_transform=None,
                 txt_name: str = "train.txt",tis_transform=None):
        self.classes= {"background":0,"cancer":1}
        self.labels = dict(zip(self.classes.values(), self.classes.keys()))
        self.root_pth = root_pth
        self.cell_path = root_pth + 'images/train/cell/'
        self.tissue_path = root_pth + 'images/train/tissue/'
        self.cell_ann_dir = root_pth + 'annotations/train/cell/'
        self.tissue_mask_dir = root_pth + 'annotations/train/tissue/'


        if cell_transform is not None:
            self.cell_transform = cell_transform
        else:
            self.cell_transform = transforms.Compose([
                transforms.ToTensor()])
        if tis_transform is not None:
            self.tis_transform = tis_transform
        else:
            self.tis_transform = transforms.Compose([
                transforms.ToTensor()])

        txt_path = root_pth+'data_cut/'+txt_name
        with open(txt_path) as read:
            self.id_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]


        self.metajson = read_json(root_pth+"metadata.json")

        # 暂时没用tissue的标注
        # assert len(self.cell_images) == len(self.tissue_images),'cell数量和tissue数量不一致'

    def __len__(self):

        return len(self.id_list)

    def __getitem__(self, idx):
        cell_image_path = os.path.join(self.cell_path, self.id_list[idx].replace(".csv", ".jpg"))
        cell_ann_path = os.path.join(self.cell_ann_dir, self.id_list[idx])
        ann = self.read_point(cell_ann_path)
        tissue_image_path = os.path.join(self.tissue_path, self.id_list[idx].replace(".csv", ".jpg"))
        tissue_mask_path = os.path.join(self.tissue_mask_dir, self.id_list[idx].replace(".csv", ".png"))
        pos_x =self.metajson['sample_pairs'][self.id_list[idx].split('.')[0]]['patch_x_offset']
        pos_y =self.metajson['sample_pairs'][self.id_list[idx].split('.')[0]]['patch_y_offset']
        pos = torch.tensor([pos_x,pos_y])
        # print("标签路径为",cell_mask_path)
        # print(self.cell_images[idx].replace(".jpg", ".png"))
        cell_image = Image.open(cell_image_path).convert('RGB')

        tissue_image = Image.open(tissue_image_path).convert('RGB')
        # 组织标注中背景为1，癌症区域为2，所以要整体减去1，背景就为0癌症区域就为1，忽略区域为254
        tissue_mask = Image.open(tissue_mask_path).convert('L')
        # /127之后 ，这样子一般细胞约为1，癌症细胞约为2

        tissue_mask = np.array(tissue_mask)-1 / 1
        tissue_mask = Image.fromarray(tissue_mask)

        mask_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512], interpolation=Image.NEAREST)])
        cell_image = self.cell_transform(cell_image)

        tissue_image = self.tis_transform(tissue_image)
        tissue_mask = mask_transform(tissue_mask)


        return {'img': cell_image, 'annot': ann}

    def read_point(self,cell_ann_path):
        try:
            df = pd.read_csv(cell_ann_path)
            if not df.empty:
                data = df.values.tolist()
                a_tensor = torch.Tensor(data)
                p1 = a_tensor[:, 0:2] - 10
                p1[p1 < 0] = 0
                p2 = a_tensor[:, 0:2] + 10
                p2[p2 > 1023] = 1023
                # print(a_tensor)
                # classes start from 0
                cls = a_tensor[:, 2].long().view(-1, 1) - 1
                return torch.cat([p1,p2,cls],dim=1).int()
            else:
                return torch.zeros((0,5))
        except pd.errors.EmptyDataError:
            return torch.zeros((0,5))

    def num_classes(self):
        return 2

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]


    def load_annotations(self, image_index):
        data = self.__getitem__(image_index)
        return np.array(data['annot'])



def read_json(fpath) -> dict:
    """This function reads a json file

    Parameters
    ----------
    fpath: Path
        path to the json file

    Returns
    -------
    dict:
        loaded data
    """
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
        

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot

    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = torch.stack(imgs,dim=0)

    return {'img': padded_imgs, 'annot': annot_padded}


class MyDataset:
    def __init__(self, root_pth = "/home/test/workspace/experiment/dataset/ocelot2023_v0.1.1/", cell_transform=None,
                 txt_name: str = "train.txt",tis_transform=None):
        self.classes= {"background":0,"cancer":1}
        self.labels = dict(zip(self.classes.values(), self.classes.keys()))
        self.root_pth = root_pth
        self.cell_path = root_pth + 'images/train/cell/'
        self.tissue_path = root_pth + 'images/train/tissue/'
        self.cell_ann_dir = root_pth + 'annotations/train/cell/'
        self.tissue_mask_dir = root_pth + 'annotations/train/tissue/'


        if cell_transform is not None:
            self.cell_transform = cell_transform
        else:
            self.cell_transform = transforms.Compose([
                transforms.ToTensor()])
        if tis_transform is not None:
            self.tis_transform = tis_transform
        else:
            self.tis_transform = transforms.Compose([
                transforms.ToTensor()])

        txt_path = root_pth+'data_cut/'+txt_name
        with open(txt_path) as read:
            self.id_list = [line.strip() for line in read.readlines() if len(line.strip()) > 0]


        self.metajson = read_json(root_pth+"metadata.json")

        # 暂时没用tissue的标注
        # assert len(self.cell_images) == len(self.tissue_images),'cell数量和tissue数量不一致'

    def __len__(self):

        return len(self.id_list)

    def __getitem__(self, idx):
        cell_image_path = os.path.join(self.cell_path, self.id_list[idx].replace(".csv", ".jpg"))
        cell_ann_path = os.path.join(self.cell_ann_dir, self.id_list[idx])
        ann = self.read_point(cell_ann_path)
        tissue_image_path = os.path.join(self.tissue_path, self.id_list[idx].replace(".csv", ".jpg"))
        tissue_mask_path = os.path.join(self.tissue_mask_dir, self.id_list[idx].replace(".csv", ".png"))
        pos_x =self.metajson['sample_pairs'][self.id_list[idx].split('.')[0]]['patch_x_offset']
        pos_y =self.metajson['sample_pairs'][self.id_list[idx].split('.')[0]]['patch_y_offset']
        pos = torch.tensor([pos_x,pos_y])
        # print("标签路径为",cell_mask_path)
        # print(self.cell_images[idx].replace(".jpg", ".png"))
        cell_image = Image.open(cell_image_path).convert('RGB')

        tissue_image = Image.open(tissue_image_path).convert('RGB')
        # 组织标注中背景为1，癌症区域为2，所以要整体减去1，背景就为0癌症区域就为1，忽略区域为254
        tissue_mask = Image.open(tissue_mask_path).convert('L')
        # /127之后 ，这样子一般细胞约为1，癌症细胞约为2

        tissue_mask = np.array(tissue_mask)-1 / 1
        tissue_mask = Image.fromarray(tissue_mask)

        mask_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512], interpolation=Image.NEAREST)])
        cell_image = self.cell_transform(cell_image)

        tissue_image = self.tis_transform(tissue_image)
        tissue_mask = mask_transform(tissue_mask)


        return {'img': cell_image, 'annot': ann / 2, 'tis_img':tissue_image, 'mask':tissue_mask,'pos': pos}

    def read_point(self,cell_ann_path):
        try:
            df = pd.read_csv(cell_ann_path)
            if not df.empty:
                data = df.values.tolist()
                a_tensor = torch.Tensor(data)
                p1 = a_tensor[:, 0:2] - 10
                p1[p1 < 0] = 0
                p2 = a_tensor[:, 0:2] + 10
                p2[p2 > 1023] = 1023
                # print(a_tensor)
                # classes start from 0
                cls = a_tensor[:, 2].long().view(-1, 1) - 1
                return torch.cat([p1,p2,cls],dim=1)
            else:
                return torch.zeros((0,5))
        except pd.errors.EmptyDataError:
            return torch.zeros((0,5))

    def num_classes(self):
        return 2

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def load_annotations(self, image_index):
        data = self.__getitem__(image_index)
        return np.array(data['annot'])


def collater_tissue(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    tis_imgs = [s['tis_img'] for s in data]
    masks = [s['mask'] for s in data]
    poses = [s['pos'] for s in data]

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot

    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = torch.stack(imgs, dim=0)
    pad_tis_imgs = torch.stack(tis_imgs, dim=0)
    pad_tis_mask = torch.stack(masks, dim=0)
    pad_pos = torch.stack(poses, dim=0)


    return {'img': padded_imgs, 'annot': annot_padded, 'tis_img':pad_tis_imgs, 'mask':pad_tis_mask, 'pos':pad_pos}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")


    dataset_train = MyDBDataset(txt_name='train.txt')
    dataset_val = MyDBDataset(txt_name='test.txt')
    print(len(dataset_train))
    print(len(dataset_val))


    data = dataset_train[2]
    # print(data['annot'])
    dataloader_train = DataLoader(dataset_train, batch_size=2, num_workers=3, collate_fn=collater)

    data_db = MyDataset()
    dataloader_db = DataLoader(data_db, batch_size=2, num_workers=3, collate_fn=collater_tissue)
    for iter_num, data in enumerate(dataloader_db):
        print(data['img'].size())
        print(data['mask'].size())
        print(data['pos'].size())

    # for iter_num, data in enumerate(dataloader_train):
    #     print(data['img'].size())
    #     print(data['annot'].int())

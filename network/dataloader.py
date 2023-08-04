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


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path       = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


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

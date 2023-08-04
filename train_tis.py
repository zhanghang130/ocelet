import argparse
import collections
import os

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import traceback

from network import model_tissue
from network.dataloader import MyDataset, CocoDataset, CSVDataset, collater_tissue, Resizer, AspectRatioBasedSampler, \
    Augmenter, Normalizer
from torch.utils.data import DataLoader


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training network.')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=200)

    parser = parser.parse_args(args)


    # using compute_mean_std.py
    mean_cell = [0.7609601, 0.57758653, 0.6961523]
    std_cell = [0.15274888, 0.18806055, 0.14399514]

    mean_tissue = [0.7652827, 0.5833039, 0.6994072]
    std_tissue = [0.15459189, 0.19640568, 0.15694112]


    import torchvision.transforms as transforms
    cell_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_cell,std=std_cell),
                                        transforms.Resize([512, 512])])
    tissue_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_tissue,std=std_tissue),
                                           transforms.Resize([512, 512])])

    dataset_train = MyDataset(cell_transform=cell_transform,tis_transform=tissue_transform)

    dataloader_train = DataLoader(dataset_train, batch_size=6, num_workers=3, collate_fn=collater_tissue)

    # Create the model

    retinanet_tis = model_tissue.resnet50(num_classes=2, pretrained=False)
    retinanet_tis.load_state_dict(torch.load('resnet50-19c8e357.pth'), strict=False)

    if torch.cuda.is_available():
        retinanet_tis = retinanet_tis.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet_tis).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet_tis)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=100)

    retinanet.train()
    # retinanet.module.freeze_bn()
    # retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(f"logs/")

    best_ap = 0
    for epoch_num in range(parser.epochs):

        retinanet.train()
        # retinanet.module.freeze_bn()
        # retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if torch.cuda.is_available():
                    img, annot = data['img'].cuda().float(), data['annot'].cuda().int()
                    tis_img, mask, pos = data['tis_img'].cuda(), data['mask'].cuda(), data['pos'].cuda()
                    (classification_loss, regression_loss), tissue_loss = retinanet([img, annot, tis_img, mask, pos])
                else:
                    print("no cuda!!!")

                # print(f"classification_loss{classification_loss}")
                # print(f"regression_loss{regression_loss}")
                #
                # print(f"tissue_loss{tissue_loss}")

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                tissue_loss=tissue_loss.mean()
                loss = classification_loss + regression_loss + tissue_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} |Tissue loss :{:1.5f} Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss),float(tissue_loss), np.mean(loss_hist)))

                del tissue_loss
                del classification_loss
                del regression_loss
            except Exception as e:
                traceback.print_exc()
                continue

        scheduler.step(np.mean(epoch_loss))

        if epoch_num % 50 == 0:
            torch.save({'model': retinanet.module.state_dict()}, f'model_tis_weight/{epoch_num}_retinanet_tis_best.pth')

    retinanet.eval()

    torch.save({'model': retinanet.module.state_dict()}, 'model_tis_weight/model_final.pt')


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    main()

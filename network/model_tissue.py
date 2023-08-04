import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms

from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet.losses import build_target,dice_loss
import torch.nn.functional as F


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6_1 = nn.ReLU()
        self.P6_2 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P6_x = self.P6_1(C5)
        P6_x = self.P6_2(P6_x)

        return [P2_x, P3_x, P4_x, P5_x, P6_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=4, num_classes=2, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class TissueHead(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 2, bilinear: bool = True, base_c: int = 64):
        super(TissueHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor):
        # x [b,c,h,w] -> x1 [b,64,h,w]
        x1 = self.in_conv(x)
        # x1 [b,64,h,w] -> x2 [b,128,h/2,w/2]
        x2 = self.down1(x1)
        # x2 [b,128,h/2,w/2] -> x3 [b,256,h/4,w/4]
        x3 = self.down2(x2)
        # x3 [b,256,h/4,w/4] -> x4 [b,512,h/8,w/8]
        x4 = self.down3(x3)

        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        tis_map = x.detach()
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)



        return {"tis_cut": tis_map,"tis_out": logits}


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # w/4,h/4
        self.layer1 = self._make_layer(block, 64, layers[0])
        # w/4,h/4
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # w/8,h/8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # w/16,h/16
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # w/32,h/32

        self.tissue_head = TissueHead(in_channels=3,num_classes=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0] - 1].conv2.out_channels, self.layer2[layers[1] - 1].conv2.out_channels,
                         self.layer3[layers[2] - 1].conv2.out_channels,self.layer4[layers[3] - 1].conv2.out_channels,]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0] - 1].conv3.out_channels, self.layer2[layers[1] - 1].conv3.out_channels,
                         self.layer3[layers[2] - 1].conv3.out_channels, self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")


        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2],fpn_sizes[3]+256)

        # input channels = 256
        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)


        self.anchors = Anchors(sizes=[8,10,12,14,16])

        # 将anchor和调节参数传入得到最后的框
        self.regressBoxes = BBoxTransform()

        # 避免框超出图片边界
        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.1

        # initialize
        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        # 冻结bn层数
        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):

        if self.training:
            img_batch, annotations, tis_cut, mask, pos  = inputs
        else:
            img_batch= inputs

        # print(f"img_batch.size():{img_batch.size()}")
        out = self.tissue_head(tis_cut)

        x = self.conv1(img_batch)
        # print(f"conv1(x).size():{x.size()}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f"maxpool(x).size():{x.size()}")

        x1 = self.layer1(x)
        # print(f"layer1(x).size():{x1.size()}")

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # print(f"layer4(x).size():{x4.size()}")
        # print(f"out['tis_cut'].size():{out['tis_cut'].size()}")

        x4 = concatenator(x4, out['tis_cut'], pos)

        features = self.fpn([x1, x2, x3, x4])
        # print(f"len fpn(x2, x3, x4):{len(features)}")

        # put P3 - P7 in regressionModel
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        # regression返回的其实是中心点的偏移和高度还有宽度的缩放:dx,dy,dw,dh


        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch).cuda()
        # print(f"anchors.size():{anchors.size()}")

        if self.training:
            weight = torch.as_tensor([1.0, 2.0]).cuda()
            return (self.focalLoss(classification, regression, anchors, annotations),
            criterion(out['tis_out'],mask.squeeze(dim=1).long(),weight))
        else:
            # 利用预测值调节anchor框
            transformed_anchors = self.regressBoxes(anchors, regression)

            # 切掉超过边框的anchor
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.to('cuda:1')
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.to('cuda:1')

            # 循环类别个数
            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                # 索引之后，变成一维了
                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]

                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


def concatenator(x, x2, pos):
    # x:[b,512,w/16,h/16] x2:[b,128,w/4,h/4]
    assert x2.size()[0] == pos.size()[0],print("batch size not right")
    batch_len = pos.size()[0]
    croped_map = []
    for i in range(batch_len):
        croped_map.append(crop_tissue(x2[i,:],pos[i,0],pos[i,1]))

    x2_cropped = torch.stack(croped_map,dim=0)
    # print(x2_cropped.shape)
    assert x2_cropped.size()[-1] == x.size()[-1],"check  crop_tissue function"
    # res = [r.squeeze() for r in x2.split(1, dim=0)]
    # print([r.size() for r in res])

    conc = torch.cat([x, x2_cropped], dim=1)
    # print(conc.shape)

    return conc


def crop_tissue(cell_image, patch_x_offset, patch_y_offset):
    x_center, y_center = int(cell_image.shape[1] * patch_x_offset), int(cell_image.shape[2] * patch_y_offset)

    ll = int(cell_image.shape[1] / 8)
    new_image = cell_image[:, x_center - ll:x_center + ll, y_center - ll:y_center + ll]


    return new_image


def criterion(inputs,  target_tissue, tissue_weight, dice: bool = True, ignore_index: int = -1):
    # print()
    loss_tissue = torch.nn.functional.cross_entropy(inputs, target_tissue, ignore_index=254, weight=tissue_weight)
    if dice is True:
        tis_dice_target = build_target(target_tissue, 2, 254)
        loss_tissue += dice_loss(inputs, tis_dice_target, multiclass=True, ignore_index=254)
    # print(f"dice loss为{loss}")
    return  loss_tissue



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    from dataloader import MyDataset,collater_tissue
    from torch.utils.data import DataLoader

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

    dataloader_train = DataLoader(dataset_train, batch_size=2, num_workers=3, collate_fn=collater_tissue)
    inputs = next(iter(dataloader_train))
    img, annot= inputs['img'].cuda().float(), inputs['annot'].cuda().int()
    tis_img, mask, pos= inputs['tis_img'].cuda(),inputs['mask'].cuda(), inputs['pos'].cuda()

    retinanet = resnet50(num_classes=2, pretrained=False)
    retinanet = retinanet.cuda()
    retinanet.training = True
    input = [img, annot,tis_img,mask, pos]
    (classification_loss, regression_loss), tissue_loss = retinanet(input)
    print(tissue_loss)
    print(f"classification_loss{classification_loss}")
    print(f"regression_loss{regression_loss}")





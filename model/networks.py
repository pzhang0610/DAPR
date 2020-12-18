from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'Discriminator', 'LocalConvBlock', 'JointSegConv', 'JointSegAllConv','JointSegDrop', 'JointSegDropGlob', 'JointSegCls','PartAtten', 'LocAtten']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False, last_stride=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

                # Zero-initialize the last BN in each residual branch,
                # so that the residual branch starts with zeros, and each residual block behaves like an identity.
                # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        y = self.layer4(x)
        return x, y


def get_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

        Layers that don't match with pretrained layers in name or size are kept unchanged.
        """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    return model_dict


def resnet18(pretrained=True, last_stride=2):
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   last_stride=last_stride)
    if pretrained:
        model_dict = get_pretrained_weights(model, model_urls['resnet18'])
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained=True, last_stride=2):
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   last_stride=last_stride)
    if pretrained:
        model_dict = get_pretrained_weights(model, model_urls['resnet34'])
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=True, last_stride=2):
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 6, 3],
                   last_stride=last_stride)
    if pretrained:
        model_dict = get_pretrained_weights(model, model_urls['resnet50'])
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=True, last_stride=2):
    model = ResNet(block=Bottleneck,
                   layers=[3, 4, 23, 3],
                   last_stride=last_stride)
    if pretrained:
        model_dict = get_pretrained_weights(model, model_urls['resnet101'])
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=True, last_stride=2):
    model = ResNet(block=Bottleneck,
                   layers=[3, 8, 36, 3],
                   last_stride=last_stride)
    if pretrained:
        model_dict = get_pretrained_weights(model, model_urls['resnet152'])
        model.load_state_dict(model_dict)
    return model


class Discriminator(nn.Module):
    def __init__(self, in_channel, classes=2):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            self.FC(in_channel=in_channel, out_channel=1024, activation=True),
            self.FC(in_channel=1024, out_channel=512, activation=True),
            self.FC(in_channel=512, out_channel=256, activation=True),
            self.FC(in_channel=256, out_channel=128, activation=True),
            self.FC(in_channel=128, out_channel=classes)
        )

        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def FC(self, in_channel, out_channel, norm=None, activation=None, dropout=None):
        layers = []
        layers.append(nn.Linear(in_channel, out_channel))
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        if norm is not None:
            layers.append(nn.BatchNorm1d(out_channel))
        if activation is not None:
            layers.append(nn.LeakyReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.discriminator(x)


def LocalConvBlock(in_channels, out_channels, bnorm=True, relu=True):
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
    if bnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class JointSegAllConv(nn.Module):
    def __init__(self, in_channel, num_features, num_classes, num_strips=6, bnneck=True):
        super(JointSegAllConv, self).__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_strips = num_strips
        self.bnneck = bnneck

        self.conv = LocalConvBlock(in_channels=4096, out_channels=self.num_features, bnorm=True, relu=True)
        for i in range(self.num_strips):
            setattr(self, 'local_conv_' + str(i), LocalConvBlock(in_channels=4096, out_channels=self.num_features, bnorm=True, relu=True))
            setattr(self, 'local_cls_' + str(i), nn.Linear(self.num_features, self.num_classes, bias=False))
            if self.bnneck:
                setattr(self, 'local_bnneck_' + str(i), nn.BatchNorm1d(self.num_features))
                getattr(self, 'local_bnneck_' + str(i)).bias.requires_grad_(False)
        if self.bnneck:
            self.glob_bnneck = nn.BatchNorm1d(self.num_features)
            self.glob_bnneck.bias.requires_grad_(False)
        self.cls = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, glob, part, norm=False):
        feat_list = {}
        logit_list = {}
        feat = []
        for i in range(self.num_strips):
            local_feat = getattr(self, 'local_conv_' + str(i))(part[i])
            local_feat = local_feat.view(local_feat.shape[0], -1)
            feat.append(self.l2norm(local_feat))
            feat_list[i] = getattr(self, 'local_bnneck_' + str(i))(local_feat)
            feat_bn = self.bnnecklayer(local_feat)
            feat_list[i] = feat_bn

            local_logit = getattr(self, 'local_cls_' + str(i))(feat_bn)
            logit_list[i] = local_logit

        glob_feat = self.conv(glob)
        glob_feat = glob_feat.view(glob_feat.shape[0], -1)
        glob_bn_feat = self.glob_bnneck(glob_feat)
        glob_logit = self.cls(glob_bn_feat)
        feat_list[self.num_strips] = glob_bn_feat
        logit_list[self.num_strips] = glob_logit

        feat.append(self.l2norm(glob_feat))
        feat = torch.cat(feat, dim=1)
        return feat, feat_list, logit_list


class JointSegConv(nn.Module):
    def __init__(self, in_channel, num_features, num_classes, num_strips=6, bnneck=True):
        super(JointSegConv, self).__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_strips = num_strips
        self.bnneck = bnneck
        self.l2norm = Normalize()
        # self.conv = LocalConvBlock(in_channels=4096, out_channels=self.num_features, bnorm=True, relu=True)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_features, kernel_size=1, padding=0,
                              bias=False)
        if self.bnneck:
            self.bnnecklayer = nn.BatchNorm1d(self.num_features)
            self.bnnecklayer.bias.requires_grad_(False)
        for i in range(self.num_strips):
            setattr(self, 'local_cls_' + str(i), nn.Linear(self.num_features, self.num_classes, bias=False))
        self.cls = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, glob, part, norm=False):
        feat_list = {}
        logit_list = {}
        feat = []
        for i in range(self.num_strips):
            local_feat = self.conv(part[i])
            local_feat = local_feat.view(local_feat.shape[0], -1)
            # feat.append(self.l2norm(local_feat))
            feat.append(local_feat)
            feat_bn = self.bnnecklayer(local_feat)
            feat_list[i] = feat_bn
            local_logit = getattr(self, 'local_cls_' + str(i))(feat_bn)
            logit_list[i] = local_logit

        glob_feat = self.conv(glob)
        glob_feat = glob_feat.view(glob_feat.shape[0], -1)

        glob_bn_feat = self.bnnecklayer(glob_feat)
        glob_logit = self.cls(glob_bn_feat)
        feat_list[self.num_strips] = glob_bn_feat
        logit_list[self.num_strips] = glob_logit

        # feat.append(self.l2norm(glob_feat))
        feat.append(glob_feat)
        feat = torch.cat(feat, dim=1)
        return feat, feat_list, logit_list


class JointSegCls(nn.Module):
    def __init__(self, in_channel, num_features, num_classes, num_strips=6, bnneck=True):
        super(JointSegCls, self).__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_strips = num_strips
        self.bnneck = bnneck
        self.l2norm = Normalize()
        self.conv = LocalConvBlock(in_channels=4096, out_channels=2048, bnorm=True, relu=True)
        self.linear = nn.Linear(in_features=2048, out_features=self.num_features, bias=False)
        if self.bnneck:
            self.bnnecklayer = nn.BatchNorm1d(self.num_features)
            self.bnnecklayer.bias.requires_grad_(False)
        for i in range(self.num_strips):
            setattr(self, 'local_cls_' + str(i), nn.Linear(self.num_features, self.num_classes, bias=False))
        self.cls = nn.Linear(in_features=self.num_features, out_features=self.num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, glob, part, norm=False):
        feat_list = {}
        logit_list = {}
        feat = []
        for i in range(self.num_strips):
            ifeat = self.conv(part[i])
            local_feat = self.linear(ifeat.view(ifeat.shape[0], -1))
            feat.append(self.l2norm(local_feat))
            feat_bn = self.bnnecklayer(local_feat)
            feat_list[i] = feat_bn
            local_logit = getattr(self, 'local_cls_' + str(i))(feat_bn)
            logit_list[i] = local_logit

        gfeat = self.conv(glob)
        glob_feat = self.linear(gfeat.view(gfeat.shape[0], -1))

        glob_bn_feat = self.bnnecklayer(glob_feat)
        glob_logit = self.cls(glob_bn_feat)
        feat_list[self.num_strips] = glob_bn_feat
        logit_list[self.num_strips] = glob_logit

        feat.append(self.l2norm(glob_feat))
        feat = torch.cat(feat, dim=1)
        return feat, feat_list, logit_list


class JointSegDrop(nn.Module):
    def __init__(self, in_channel, num_features, num_classes, num_strips=6, bnneck=True):
        super(JointSegDrop, self).__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_strips = num_strips
        self.bnneck = bnneck
        self.l2norm = Normalize()
        # self.conv_rd = LocalConvBlock(in_channels=4096, out_channels=2048, bnorm=True, relu=True)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_features, kernel_size=1, padding=0, bias=False)
        self.bnnecklayer = nn.BatchNorm1d(num_features=self.num_features)
        for i in range(self.num_strips):
            setattr(self, 'local_cls_' + str(i), nn.Linear(self.num_features, self.num_classes, bias=False))
        self.glob_cls = nn.Linear(self.num_features, self.num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, norm=False):
        # x = self.conv_rd(x)
        x = self.conv(x).squeeze()
        bnx = self.bnnecklayer(x)

        logit_list = {}
        for i in range(self.num_strips):
            local_i = bnx[:, :, i].squeeze()
            local_logit = getattr(self, 'local_cls_' + str(i))(local_i)
            logit_list[i] = local_logit
        feat = self.l2norm(x).view(x.size(0), -1)
        bn_feat = self.l2norm(bnx).view(bnx.size(0), -1)
        # glob_feat = self.l2norm(F.adaptive_avg_pool1d(x, 1))
        # glob_bnfeat = self.l2norm(F.adaptive_avg_pool1d(bnx, 1))
        return feat, bn_feat, logit_list


class JointSegDropGlob(nn.Module):
    def __init__(self, in_channel, num_features, num_classes, num_strips=6, bnneck=True):
        super(JointSegDropGlob, self).__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_strips = num_strips
        self.bnneck = bnneck
        self.l2norm = Normalize()
        # self.conv_rd = LocalConvBlock(in_channels=4096, out_channels=2048, bnorm=True, relu=True)
        self.conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.num_features, kernel_size=1, padding=0, bias=False)
        self.bnnecklayer = nn.BatchNorm1d(num_features=self.num_features)
        for i in range(self.num_strips):
            setattr(self, 'local_cls_' + str(i), nn.Linear(self.num_features, self.num_classes, bias=False))
        self.glob_cls = nn.Linear(self.num_features, self.num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x, g, norm=False):
        # x = self.conv_rd(x)
        # pdb.set_trace()
        x = self.conv(x).squeeze()
        bnx = self.bnnecklayer(x)
        g = self.conv(g).squeeze(3)
        bng = self.bnnecklayer(g)

        logit_list = {}
        for i in range(self.num_strips):
            local_i = bnx[:, :, i].squeeze()
            local_logit = getattr(self, 'local_cls_' + str(i))(local_i)
            logit_list[i] = local_logit
        feat = torch.cat((x, g), dim=2) #
        feat = self.l2norm(feat).view(feat.size(0), -1)
        bn_feat = torch.cat((bnx, bng), dim=2)
        bn_feat = self.l2norm(bn_feat).view(bn_feat.size(0), -1)

        glob_logit = self.glob_cls(bng.squeeze())
        logit_list[self.num_strips] = glob_logit
        # feat = self.l2norm(x).view(x.size(0), -1)
        # bn_feat = self.l2norm(bnx).view(bnx.size(0), -1)


        # glob_feat = self.l2norm(F.adaptive_avg_pool1d(x, 1))
        # glob_bnfeat = self.l2norm(F.adaptive_avg_pool1d(bnx, 1))
        return feat, bn_feat, logit_list

class GlobSeg(nn.Module):
    def __init__(self, in_channel, num_features, num_classes):
        super(GlobSeg, self).__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes

        self.glob_conv = LocalConvBlock(in_channels=self.in_channel, out_channels=self.num_features, bnorm=True, relu=True)
        self.glob_cls = nn.Linear(in_features=self.num_features, out_features=self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        feat =self.glob_conv(x)
        feat = feat.view(feat.shape[0], -1)
        logit = self.glob_cls(feat)
        return feat, logit


class PartSeg(nn.Module):
    def __init__(self, in_channel, num_features, num_classes, num_strips=6):
        super(PartSeg, self).__init__()
        self.in_channel = in_channel
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_strips = num_strips
        # pdb.set_trace()

        for i in range(self.num_strips):
            setattr(self, 'local_conv_' + str(i), LocalConvBlock(in_channels=self.in_channel, out_channels=self.num_features, bnorm=True, relu=True))
            setattr(self, 'local_cls_' + str(i), nn.Linear(in_features=self.num_features, out_features=self.num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        x: {batch_size * num_features * 1 * 1}, length = 6
        """
        feat_list = {}
        logit_list = {}
        pdb.set_trace()
        for i in range(self.num_strips):
            local_feat = getattr(self, 'local_conv_' + str(i))(x[i])
            local_feat = local_feat.view(local_feat.shape[0], -1)
            feat_list[i] = local_feat
            local_logit = getattr(self, 'local_cls_' + str(i))(local_feat)
            logit_list[i] = local_logit
        return feat_list, logit_list


class PartSegConv(nn.Module):
    def __init__(self, num_features, num_classes, num_strips=6):
        super(PartSeg, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_strips = num_strips

        self.local_conv = LocalConvBlock(in_channels=2048, out_channels=self.num_features, bnorm=True, relu=True)
        for i in range(self.num_strips):
            setattr(self, 'local_cls_' + str(i), nn.Linear(self.num_features, self.num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

    # def LocalConvBlock(self, in_channels, out_channels, bnorm=True, relu=True):
    #     layers = []
    #     layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False))
    #     if bnorm:
    #         layers.append(nn.BatchNorm2d(out_channels))
    #     if relu:
    #         layers.append(nn.ReLU(inplace=True))
    #     return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: {batch_size * num_features * 1 * 1}, length = 6
        """
        feat_list = {}
        logit_list = {}
        pdb.set_trace()
        for i in range(self.num_strips):
            local_feat = self.local_conv(x[i])
            local_feat = local_feat.view(local_feat[0], -1)
            feat_list[i] = local_feat
            local_logit = getattr(self, 'local_cls_' + str(i))(local_feat)
            logit_list[i] = local_logit
        return feat_list, logit_list


class LocAtten(nn.Module):
    def __init__(self, in_channel):
        super(LocAtten, self).__init__()
        self.in_channel = in_channel

        self.query_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=self.in_channel, out_channels=self.in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x: input feature maps (Batch_size * Channel * W * H)
        returns:
            out: self attention value + input feature
            """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class ChnAtten(nn.Module):
    def __init__(self):
        super(ChnAtten, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
          inputs:
              x: input feature maps (Batch_size * Channel * W * H)
          returns:
              out: self attention value + input feature
              """
        m_batchsize, C, width, height = x.size()
        proj_query = x.view(m_batchsize, C, -1) # B X C X (N)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) # B X (N) X C
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class PartAtten(nn.Module):
    def __init__(self, in_channel, num_strips):
        super(PartAtten, self).__init__()
        self.in_channel = in_channel
        self.num_strips = num_strips

        self.query = nn.Linear(in_features=self.in_channel, out_features=self.in_channel//8, bias=False)
        self.key = nn.Linear(in_features=self.in_channel, out_features=self.in_channel // 8, bias=False)
        self.value = nn.Linear(in_features=self.in_channel, out_features=self.in_channel, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
        x: dictionary to store each part with shape B * C * 1 * 1
        """
        m_batchsize, C, width, height = x[0].size()
        feat = []
        for i in range(self.num_strips):
            feat.append(torch.squeeze(x[i]))
        feat = torch.stack(feat, dim=1) # B * strips * C
        proj_query = self.query(feat.view(-1, C)).view(m_batchsize, self.num_strips, -1)
        proj_key = self.key(feat.view(-1, C)).view(m_batchsize, self.num_strips, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # BX (4) X (4)

        proj_value = self.value_conv(feat.view(-1, C)).view(m_batchsize, self.num_strips, -1).permute(0, 2, 1) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, self.num_strips).permute(0, 2, 1)

        out = self.gamma * out + feat
        return out, attention


if __name__ == "__main__":
    # from netutils import *
    # x = torch.randn(64, 3, 288, 144)
    # model = resnet50(pretrained=True, last_stride=2)
    # print_network(model)
#     mid, out = model(x)
#     print(mid.shape)
#     print(out.shape)
#     g = make_dot(out)
#     g.view()
#     discriminator = Discriminator(in_channel=2056, classes=2)
#     for m in discriminator.parameters():
#         print(m)
    PCB = PartSeg(num_features=256, num_classes=1000, num_strips=6)
    x = {}
    for i in range(6):
        x[i] = torch.randn(size=(64, 2048, 1, 1))
    feat, logits = PCB(x)

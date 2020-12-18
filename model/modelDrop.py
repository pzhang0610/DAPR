from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import *
from .netutils import ReverseLayer
import pdb


class SDAN(nn.Module):
    def __init__(self, num_strips, num_features, num_classes, drop=0):
        super(SDAN, self).__init__()
        self.num_strips = num_strips
        self.num_features = num_features
        self.num_classes = num_classes

        self.base_rgb = resnet50(pretrained=True, last_stride=1)
        self.base_ir = resnet50(pretrained=True, last_stride=1)
        self.discriminator = Discriminator(in_channel=2048//2*5, classes=2)
        self.atten_rgb =LocAtten(in_channel=2048)
        self.atten_ir = LocAtten(in_channel=2048)
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(drop)
        # self.part_seg = PartSeg(in_channel=4096, num_features=self.num_features, num_classes=self.num_classes, num_strips=self.num_strips)
        # self.glob_seg = GlobSeg(in_channel=4096, num_features=self.num_features, num_classes=self.num_classes)
        self.conv_rd = LocalConvBlock(in_channels=2048, out_channels=1024, bnorm=True, relu=True)
        self.joint_seg = JointSegDropGlob(in_channel=1024, num_features=self.num_features, num_classes=self.num_classes, num_strips=self.num_strips, bnneck=True)

    def forward(self, rgb_img, ir_img, alpha, out_feature=False, norm=False):
        rgb_map, rgb_feat = self.base_rgb(rgb_img)
        ir_map, ir_feat = self.base_ir(ir_img)
        rgb_atten = self.atten_rgb(rgb_feat)
        ir_atten = self.atten_ir(ir_feat)

        rgb_atten_feat = F.adaptive_avg_pool2d(rgb_atten, output_size=1)
        ir_atten_feat = F.adaptive_avg_pool2d(ir_atten, output_size=1)
        aligned_feat = rgb_atten_feat - ir_atten_feat

        rgb_map = torch.cat((rgb_map, rgb_feat, rgb_atten), dim=1)
        ir_map = torch.cat((ir_map, ir_feat, ir_atten), dim=1)
        map = torch.cat((rgb_map, ir_map), dim=0)
        map = F.adaptive_avg_pool2d(map, output_size=1).squeeze()
        if not out_feature:
            reverse_map = ReverseLayer.apply(map, alpha)
            domain_logit = self.discriminator(reverse_map)

        # rgb_feat = torch.cat((rgb_feat, rgb_atten), dim=1)
        # ir_feat = torch.cat((ir_feat, ir_atten), dim=1)
        rgb_feat = rgb_atten
        ir_feat = ir_atten
        all_feat = torch.cat((rgb_feat, ir_feat), dim=0)
        all_feat = self.conv_rd(all_feat)
        # all_feat = F.adaptive_avg_pool2d(all_feat, (self.num_strips, 1))
        # all_feat =  F.adaptive_max_pool2d(all_feat, (self.num_strips, 1))
        all_feat = F.adaptive_avg_pool2d(all_feat, (self.num_strips, 1)) # + F.adaptive_max_pool2d(all_feat, (self.num_strips, 1))#
        glob_feat = F.adaptive_avg_pool2d(all_feat, output_size=1) # + F.adaptive_max_pool2d(all_feat, output_size=1) #

        if self.drop:
            all_feat = self.dropout(all_feat)
            glob_feat = self.dropout(glob_feat)

        feat, bn_feat, logit_list = self.joint_seg(all_feat, glob_feat, norm=norm)
        if out_feature:
            return bn_feat
        else:
            return domain_logit, feat, aligned_feat, bn_feat, logit_list






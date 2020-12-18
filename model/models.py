from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .networks import *
from .netutils import ReverseLayer


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
        self.joint_seg = JointConvSeg(in_channel=4096, num_features=self.num_features, num_classes=self.num_classes, num_strips=self.num_strips, bnneck=True)

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

        rgb_feat = torch.cat((rgb_feat, rgb_atten), dim=1)
        ir_feat = torch.cat((ir_feat, ir_atten), dim=1)

        part_feat_list = {}
        N, C, H, W = rgb_feat.shape
        strip_height = H // self.num_strips
        for i in range(self.num_strips):
            rgb_feat_parti = F.adaptive_avg_pool2d(rgb_feat[:, :, i * strip_height:(i+1) * strip_height, :], output_size=1)
            ir_feat_parti = F.adaptive_avg_pool2d(ir_feat[:, :, i * strip_height:(i+1) * strip_height, :], output_size=1)
            part_feat_i = torch.cat((rgb_feat_parti, ir_feat_parti), dim=0)
            if self.drop:
                part_feat_i = self.dropout(part_feat_i)
            part_feat_list[i] = part_feat_i

        rgb_glob_feat = F.adaptive_avg_pool2d(rgb_feat, output_size=1)
        ir_glob_feat = F.adaptive_avg_pool2d(ir_feat, output_size=1)
        glob_feat = torch.cat((rgb_glob_feat, ir_glob_feat))
        if self.drop:
            glob_feat = self.dropout(glob_feat)
        feat, feat_list, logit_list = self.joint_seg(glob_feat, part_feat_list, norm=norm)
        if out_feature:
            return feat_list
        else:
            return domain_logit, feat, aligned_feat, feat_list, logit_list






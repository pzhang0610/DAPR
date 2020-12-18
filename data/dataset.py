from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from PIL import Image
import torch.utils.data as torData
import pdb

def read_image(img_path, width, height):
    """Keep read image until succeed."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path)
            img = img.resize((width, height), Image.ANTIALIAS)
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
        return img


class ImageDataset(torData.Dataset):
    def __init__(self, rgb_source, ir_source, width, height, transform=None):
        self.rgb_src = rgb_source
        self.ir_src = ir_source
        self.transform = transform
        self.width = width
        self.height = height

    def __getitem__(self, index):
        rgb_path, rgb_pid, rgb_camid = self.rgb_src[index[0]]
        ir_path, ir_pid, ir_camid = self.ir_src[index[1]]

        rgb_img = read_image(rgb_path, self.width, self.height)
        ir_img = read_image(ir_path, self.width, self.height)

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            ir_img = self.transform(ir_img)
        return rgb_img, rgb_pid, rgb_camid, ir_img, ir_pid, ir_camid

    def __len__(self):
        return len(self.rgb_src)


class TestDataset(torData.Dataset):
    def __init__(self, data_source, width, heigh, transform=None):
        self.data_src = data_source
        self.width = width
        self.height = heigh
        self.transform = transform

    def __getitem__(self, index):
        img_path, img_pid, img_camid = self.data_src[index]
        # print(img_pid, img_camid)
        img = read_image(img_path, self.width, self.height)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_pid, img_camid

    def __len__(self):
        return len(self.data_src)


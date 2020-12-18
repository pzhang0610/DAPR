from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import torch
import torch.nn as nn
from data import init_dataset
from data.sampler import IdentitySampler, RandomIdentitySampler
from data import ImageDataset, TestDataset
from torch.utils.data import DataLoader
from model import SDAN

from argparse import ArgumentParser
from datetime import datetime
import time
import torch.backends.cudnn as cudnn
import numpy as np
import shutil

from metrics import compute_distance_matrix, evaluate

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tools import Logger, str_boolean, AverageMeter
import pdb


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_transformer = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_src = init_dataset(name=args.dataset, mode=args.eval_mode, shot=args.num_shot)

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SDAN(num_strips=args.num_strips, num_features=args.num_feat, num_classes=args.num_classes,
                 drop=args.dropout)
    model = nn.DataParallel(model).to(devices)
    if args.mode == 'eval':
        print('{} Evaluating...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        checkpoint = torch.load(osp.join(args.save_path, 'best_model.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        rank1 = eval(model, data_src, test_transformer, args.max_epoch, args, ranks=[1, 5, 10, 20])
        return rank1


def eval(model, data_src, transformer, epoch, args, ranks=[1, 5, 10, 20]):
    model.eval()
    query_loader = DataLoader(TestDataset(data_src.query_imgs, width=args.img_width, heigh=args.img_height,
                                            transform=transformer), batch_size=args.test_batch, shuffle=False,
                                num_workers=args.num_workers)  # ir
    print("{}: Extracting probe feature...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    qf, qf_glob, q_pids, q_camids = [], [], [], []
    start_time = time.time()
    with torch.no_grad():
        for idx, (img, labels, camid) in enumerate(query_loader):
            img = img.float().cuda()
            feat_list = model(img, img, alpha=None, out_feature=True)
            feature = torch.cat(tuple(feat_list.values()), dim=1)[1]
            qf_glob.append(feat_list[args.num_strips][1].detach().cpu())
            qf.append(feature.detach().cpu())
            q_pids.extend(labels)
            q_camids.extend(camid.numpy())
        qf = torch.stack(qf)
        qf_glob = torch.stack(qf_glob)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    print('Extracting Time:\t {:.3f}'.format(time.time() - start_time))

    mAPs = []
    Accus = []
    for i in range(10):
        gallery_loader = DataLoader(TestDataset(data_src._process_gallery_data(mode='all', shot='single')[0], width=args.img_width, heigh=args.img_height,
                                                transform=transformer), batch_size=args.test_batch, shuffle=False,
                                    num_workers=args.num_workers)  # rgb
        print("{}: Extracting gallery feature...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        gf, gf_glob, g_pids, g_camids = [], [], [], []
        start_time = time.time()
        with torch.no_grad():
            for idx, (img, labels, camid) in enumerate(gallery_loader):
                # pdb.set_trace()
                img = img.float().cuda()
                feat_list = model(img, img, alpha=None, out_feature=True)
                feature = torch.cat(tuple(feat_list.values()), dim=1)[0]
                # pdb.set_trace()
                gf_glob.append(feat_list[args.num_strips][0].detach().cpu())
                gf.append(feature.detach().cpu())
                g_pids.extend(labels)
                g_camids.extend(camid.numpy())

            gf = torch.stack(gf)
            gf_glob = torch.stack(gf_glob)
            g_pids = np.asarray(g_pids)
            g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print('Extracting Time:\t {:.3f}'.format(time.time() - start_time))

        print("Computing distance matrix")
        if args.metric == 'euclidean':
             distmat = compute_distance_matrix(qf, gf, metric='euclidean')
             glob_distmat = compute_distance_matrix(qf_glob, gf_glob, metric='euclidean')
        elif args.metric == 'cosine':
            distmat = compute_distance_matrix(qf, gf, metric='cosine')
            glob_distmat = compute_distance_matrix(qf_glob, gf_glob, metric='cosine')
        else:
            raise ValueError("Type of distance {} is not supported. Pls select from euclidean and cosine...".format(args.metric))

        cmc, mAP, mINP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20, dataset=args.dataset)
        cmc_glob, mAP_glob, mINP_glob = evaluate(glob_distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20, dataset=args.dataset)

        print("Epoch: [{}] Results ----------".format(epoch + 1))
        print("mAP: {:.1%}, GmAP: {:.1%}".format(mAP, mAP_glob))
        print("mINP: {:.1%}, GmINP: {:.1%}".format(mINP, mINP_glob))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}, {:.1%}".format(r, cmc[r - 1], cmc_glob[r -1]))
        print("------------------")
        mAPs.append(mAP)
        Accus.append(cmc[0])
    print('Mean mAP: {:.1%}'.format(np.mean(np.array(mAP))))
    print("Mean Accuracy: {:.1%}".format(np.mean(np.array(Accus))))

    return cmc[0]

if __name__ == "__main__":
    parser = ArgumentParser(description="Parameters for SDAN")
    parser.add_argument('--mode', type=str, default='train', help='executive mode, train, resume or eval')
    parser.add_argument('--dataset', type=str, default='sysu', help='the used dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # image related parameters
    parser.add_argument('--img_height', type=int, default=288, help='image height')
    parser.add_argument('--img_width', type=int, default=144, help='image width')
    parser.add_argument('--num_strips', type=int, default=6, help='number of horizonal body strips')
    parser.add_argument('--num_feat', type=int, default=512, help='number of features')
    parser.add_argument('--num_classes', type=int, default=395, help='number of training classes')
    parser.add_argument('--erasing_p', type=float, default=0.5, help='possibility of random erasing')
    parser.add_argument('--trail', type=int, default=1, help='number of trails for RegDB cross-validation, 1,...,10')

    # training parameters
    parser.add_argument('--sampler', type=str, default='identity', help='sampler is used, identity or random')
    parser.add_argument('--train_batch', type=int, default=64, help='number of images in a batch')
    parser.add_argument('--num_instance', type=int, default=4, help='number of images per person in a batch')
    parser.add_argument('--optim', type=str, default='sgd', help='optimizer is used, sgd or adam')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate， 0。00035 for Adam, 0.1 for sgd')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers to load data')
    parser.add_argument('--max_epoch', type=int, default=100, help='maximum training epochs')

    # parameters for loss
    parser.add_argument('--margin', type=float, default=0.5, help='margin of triplet loss')
    parser.add_argument('--dropout', type=float, default=0, help='dropout possibility')
    parser.add_argument('--triplet', type=str, default='hard', help='type of triplet loss')
    parser.add_argument('--label_smooth', type=str, default='true', help='whether label smooth is used')
    parser.add_argument('--align_norm', type=str, default='fro', help='normalization manner, select from fro and abs')
    parser.add_argument('--center', type=str, default='false', help='whether center loss is used')
    parser.add_argument('--norm', type=str, default='false', help='whether feature is l2 normalized')

    # eval parameters
    parser.add_argument('--test_batch', type=int, default=1, help='batch size for testing, default is 1')
    parser.add_argument('--eval_step', type=int, default=10, help='number of epoch to perform evaluation')
    parser.add_argument('--num_shot', type=str, default='single', help='number of shots for evaluation, single or all')
    parser.add_argument('--eval_mode', type=str, default='all', help='evaluation mode: all and indoor search')

    parser.add_argument('--metric', type=str, default='cosine', help='evaluation distance, euclidean or cosine')
    # parameters for model saving
    parser.add_argument('--save_path', type=str, default='./logs/0512/', help='the path to save model')
    parser.add_argument('--ckpt_name', type=str, default='ckpt_epoch-100_sampler-identity.pth.tar')
    parser.add_argument('--log_path', type=str, default='./logs/0512/', help='the path to store logs')
    parser.add_argument('--vis_log_dir', type=str, default='./logs/0512/', help='the path to store visualization log')
    args = parser.parse_args()
    main(args)
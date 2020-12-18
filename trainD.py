from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import torch
import torch.nn as nn
from data import init_dataset
from data.sampler import IdentitySampler, RandomIdentitySampler
from data import ImageDataset, TestDataset, RandomErasing
from torch.utils.data import DataLoader
from model.modelDrop import SDAN
from model.netutils import WarmupMultiStepLR
from loss import SmoothCrossEntropyLoss
from loss import BatchHardTripletLoss, CenterLoss,TripletLoss_WRT, BiBatchHardTripletLoss

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
    log_path = osp.join(args.log_path, args.dataset, 'log.txt')
    logger = Logger(logger_path=log_path)
    logger("{} Begin training...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    logger(args)

    writer = SummaryWriter(args.vis_log_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # Load data, produce DataLoader
    logger("{} Loading data...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    train_transformer = transforms.Compose([
        transforms.RandomCrop((args.img_height, args.img_width), padding=10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
        # transforms.RandomGrayscale(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=args.erasing_p, mean=[0.485, 0.456, 0.406])
    ])

    test_transformer = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_src = init_dataset(name=args.dataset, mode=args.eval_mode, shot=args.num_shot)
    # select sampler method, identity_sampler selects batch_size person, each person with 1 images/modality, random
    # sampler select batch_size person and each person with batch_instance images/modality.
    if args.sampler == 'identity':
        # sampler = IdentitySampler(data_src.train_rgb_imgs, data_src.train_ir_imgs, batch_size=args.train_batch)
        sampler = IdentitySampler(data_src.train_rgb_imgs, data_src.train_ir_imgs, batch_size=args.train_batch, batch_instance=args.num_instance)
    elif args.sampler == 'random':
        sampler = RandomIdentitySampler(data_src.train_rgb_imgs, data_src.train_ir_imgs, batch_pid=args.train_batch, batch_instance=args.num_instance)
    else:
        raise ValueError("Sampler {} is not support. Please select from 'identity' and '' random...".format(args.sampler))

    # data loader
    train_loader = DataLoader(ImageDataset(data_src.train_rgb_imgs, data_src.train_ir_imgs, width=args.img_width,
                                           height=args.img_height, transform=train_transformer),
                              batch_size=args.train_batch, sampler=sampler,
                              num_workers=args.num_workers, drop_last=True)
    test_rgb_loader = DataLoader(TestDataset(data_src.gallery_imgs, width=args.img_width, heigh=args.img_height,
                                            transform=test_transformer), batch_size=args.test_batch, shuffle=False,
                                 num_workers=args.num_workers)  # rgb
    test_ir_loader = DataLoader(TestDataset(data_src.query_imgs, width=args.img_width, heigh=args.img_height,
                                          transform=test_transformer), batch_size=args.test_batch, shuffle=False,
                                num_workers=args.num_workers)  # ir

    cudnn.benchmark = True
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SDAN(num_strips=args.num_strips, num_features=args.num_feat, num_classes=args.num_classes, drop=args.dropout)
    model = nn.DataParallel(model).to(devices)
    if args.mode =='eval':
        logger('{} Evaluating...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        checkpoint = torch.load(osp.join(args.save_path, 'best_model.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        rank1 = eval(model, test_rgb_loader, test_ir_loader, logger, args.max_epoch, args, ranks=[1, 5, 10, 20])
        return rank1

    elif args.mode == 'resume':
        logger('{} Resume training...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        checkpoint = torch.load(osp.join(args.save_path, args.ckpt_name))
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = int(args.ckpt_name.split('_')[1].split('-')[1])

    elif args.mode == 'train':
        start_epoch = 0

    else:
        raise ValueError("{} The mode '{}' does not support, please select from train, resume and eval...".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.mode))

# Define losses
    criterion = {}
    sxent = SmoothCrossEntropyLoss(num_classes=args.num_classes)
    if args.triplet == 'hard':
        tripletloss = BatchHardTripletLoss(margin=args.margin)
    elif args.triplet == 'soft':
        tripletloss = TripletLoss_WRT()
    else:
        raise ValueError("{} is not support".format(args.triplet))

    bitripletloss = BiBatchHardTripletLoss(margin=args.margin)
    xent = nn.CrossEntropyLoss()

    criterion['sxent'] = sxent.cuda()
    criterion['triplet'] = tripletloss.cuda()
    criterion['xent'] = xent.cuda()
    criterion['bitriplet'] = bitripletloss.cuda()
    if str_boolean(args.center):
        logger("{}: Center loss is adopted...".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        centerloss = CenterLoss(num_classes=args.num_classes, feat_dim=args.num_feat * (args.num_strips + 1))
        criterion['center'] = centerloss.cuda()

# Define optimizer
    base_param_ids = list(map(id, model.module.base_rgb.parameters())) + list(map(id, model.module.base_ir.parameters()))
    base_params = list(model.module.base_rgb.parameters()) + list(model.module.base_ir.parameters())
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]

    param_groups = [{'params': base_params, 'lr_multi': 0.1},
                    {'params': new_params, 'lr_multi': 1.0}]

    optimizers = {}
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay,
                                    momentum=0.9, nesterov=True)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError('{} Optimizer {} does not support...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                      args.optim))
    optimizers['optim'] = optimizer
    if str_boolean(args.center):
        center_optimizer = torch.optim.SGD(centerloss.parameters(), lr=0.5)
        optimizers['c_optim'] = center_optimizer

    best_accu = 0
    for epoch in range(start_epoch, args.max_epoch):
        train(model, train_loader, optimizers, criterion, logger, writer, epoch=epoch, args=args)
        if (epoch+1) % args.eval_step == 0 or epoch + 1 == args.max_epoch:
            logger('{}: Performing evaluation...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            accu = eval(model, gallery_loader=test_rgb_loader, query_loader=test_ir_loader, logger=logger, epoch=epoch,
                        args=args, ranks=[1, 5, 10, 20])
            is_best = accu > best_accu
            if is_best: best_accu = accu
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            ckpt_name = os.path.join(args.save_path, 'ckpt_epoch-{}_sampler-{}.pth.tar'.format(epoch + 1, args.sampler))
            state_dict = model.state_dict()
            torch.save({'state_dict':state_dict, 'epoch': epoch + 1}, ckpt_name)
            if is_best:
                shutil.copyfile(ckpt_name, os.path.join(args.save_path, 'best_model.pth.tar'))
            logger('{}: Best Accu: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), best_accu))
    writer.close()


# def adjust_lr(optimizer, epoch):
#     if epoch <= 30:
#         lr = args.lr
#     elif 30 < epoch <= 50:
#         lr = args.lr * 0.1
#     else:
#         lr = args.lr * 0.01
#     for g in optimizer.param_groups:
#         g['lr'] = lr * g.get('lr_multi', 1.0)

def adjust_lr(optimizer, epoch):
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 20:
        lr = args.lr
    elif 20 <= epoch < 50:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_multi', 1.0)


def adjust_lr_adam(optimizer, epoch):
    if epoch <= 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 < epoch <= 40:
        lr = args.lr
    elif 40 < epoch <= 70:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_multi', 1.0)

# def warmup_lr_scheduler(optimizer, milestones, gamma=0.1, warmup_factor=1.0/3, warmup_iters=500, warmup_method='linear', last_epoch=-1):
#     scheduler = WarmupMultiStepLR(optimizer, milestones, gamma=gamma, warmup_factor=warmup_factor, warmup_iters=warmup_iters,
#                                   warmup_method=warmup_method, last_epoch=last_epoch)
#     return scheduler


def train(model, data_loader, optimizers, criterions, logger, writer, epoch, args):
    adjust_lr(optimizers['optim'], epoch)
    model.train()
    logger("{} Epoch {}/{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, args.max_epoch))
    TlossMeter = AverageMeter()
    IDlossMeter = AverageMeter()
    DAlossMeter =AverageMeter()
    AGlossMeter = AverageMeter()
    TPlossMeter = AverageMeter()
    i = 0

    for idx, (rgb_img, rgb_labels, rgb_camid, ir_img, ir_labels, ir_camid) in enumerate(data_loader):
        p = float(i + epoch * len(data_loader)) / args.max_epoch / len(data_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        rgb_img = rgb_img.float().cuda()
        ir_img = ir_img.float().cuda()
        rgb_labels = rgb_labels.long().cuda()
        ir_labels = ir_labels.long().cuda()
        labels = torch.cat((rgb_labels, ir_labels))
        domain_label = torch.cat((torch.ones_like(rgb_labels), torch.zeros_like(ir_labels))).long().cuda()
        domain_logit, feat, aligned_feat, bn_feat, logit_list = model(rgb_img, ir_img, alpha, out_feature=False, norm=str_boolean(args.norm))
        if str_boolean(args.label_smooth):
            id_criterion = criterions['sxent']
        else:
            id_criterion = criterions['xent']

        part_id_loss = sum([id_criterion(logit_list[idx], labels) for idx in range(args.num_strips)])/ args.num_strips
        glob_id_loss = id_criterion(logit_list[args.num_strips], labels)
        # id_loss = part_id_loss
        id_loss = 0.5 * part_id_loss + 0.5 * glob_id_loss

        if args.align_norm == 'fro':
            aligned_loss = torch.sum(torch.norm(aligned_feat, p='fro', dim=1, keepdim=False))/args.train_batch
        elif args.align_norm == 'abs':
            aligned_loss = torch.sum(torch.norm(aligned_feat, p=1, dim=1, keepdim=False))/args.train_batch
        else:
            raise ValueError("Wrong input for the normalization, Pls select from 'fro' and 'abs'...")

        domain_loss = criterions['xent'](domain_logit, domain_label)
        rgb_triplet_loss = criterions['triplet'](feat[:args.train_batch, :], rgb_labels)
        ir_triplet_loss = criterions['triplet'](feat[args.train_batch:, :], ir_labels)
        bitripletloss = criterions['bitriplet'](feat, labels)
        # if str_boolean(args.center):
        #     center_loss = criterions['center'](feat, labels)
        #     embed_loss = triplet_loss + 0.0005 * center_loss
        # else:
        #     embed_loss = triplet_loss

        # loss = id_loss + domain_loss + 0.5*(rgb_triplet_loss + ir_triplet_loss) + bitripletloss + 0.0001 * aligned_loss
        loss = id_loss + domain_loss + bitripletloss #+ 0.1 * (rgb_triplet_loss + ir_triplet_loss) #+ 0.0001 * aligned_loss

        optimizers['optim'].zero_grad()
        loss.backward()
        optimizers['optim'].step()

        if str_boolean(args.center):
            optimizers['c_optim'].zero_grad()
            for param in criterions['center'].parameters():
                param.grad.data *= (1. / 0.0005)
            optimizers['c_optim'].step()

        # update loss
        TlossMeter.update(loss, args.train_batch)
        DAlossMeter.update(domain_loss, args.train_batch)
        TPlossMeter.update(bitripletloss, args.train_batch)
        AGlossMeter.update(aligned_loss, args.train_batch)
        IDlossMeter.update(id_loss, args.train_batch)

        if idx % 50 == 0:
            logger("{}: Epoch: [{}/{}], Iter: [{}/{}] ID_loss: {:.5f}, Tri_loss: {:.5f}, DA_loss: {:.5f}, "
                   "Align_loss: {:.5f}, Tot_loss: {:.5f}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                 epoch + 1, args.max_epoch, idx, len(data_loader),
                                                                 id_loss, bitripletloss, domain_loss, aligned_loss, loss))

    writer.add_scalar('total_loss', TlossMeter.avg, epoch + 1)
    writer.add_scalar('identity_loss', IDlossMeter.avg, epoch + 1)
    writer.add_scalar('triplet_loss', TPlossMeter.avg, epoch + 1)
    writer.add_scalar('align_loss', AGlossMeter.avg, epoch + 1)
    writer.add_scalar('domain_loss', DAlossMeter.avg, epoch + 1)


def eval(model, gallery_loader, query_loader, logger, epoch, args, ranks=[1, 5, 10, 20]):
    model.eval()
    logger("{}: Extracting gallery feature...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    gf, gf_glob, g_pids, g_camids = [], [], [], []
    start_time = time.time()
    with torch.no_grad():
        for idx, (img, labels, camid) in enumerate(gallery_loader):
            # pdb.set_trace()
            img = img.float().cuda()
            bn_feat = model(img, img, alpha=None, out_feature=True, norm=str_boolean(args.norm))
            feature = bn_feat[0]
            # pdb.set_trace()
            gf.append(feature.detach().cpu())
            g_pids.extend(labels)
            g_camids.extend(camid.numpy())

        gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
    logger("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    logger('Extracting Time:\t {:.3f}'.format(time.time() - start_time))

    logger("{}: Extracting probe feature...".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    qf, qf_glob, q_pids, q_camids = [], [], [], []
    start_time = time.time()
    with torch.no_grad():
        for idx, (img, labels, camid) in enumerate(query_loader):
            img = img.float().cuda()
            bn_feat = model(img, img, alpha=None, out_feature=True, norm=str_boolean(args.norm))
            feature = bn_feat[1]
            qf.append(feature.detach().cpu())
            q_pids.extend(labels)
            q_camids.extend(camid.numpy())
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
    logger("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    logger('Extracting Time:\t {:.3f}'.format(time.time() - start_time))

    logger("Computing distance matrix")
    if args.metric == 'euclidean':
         distmat = compute_distance_matrix(qf, gf, metric='euclidean')
    elif args.metric == 'cosine':
        distmat = compute_distance_matrix(qf, gf, metric='cosine')
    else:
        raise ValueError("Type of distance {} is not supported. Pls select from euclidean and cosine...".format(args.metric))

    cmc, mAP, mINP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20, dataset=args.dataset)

    logger("Epoch: [{}] Results ----------".format(epoch + 1))
    logger("mAP: {:.1%}".format(mAP))
    logger("mINP: {:.1%}".format(mINP))
    logger("CMC curve")
    for r in ranks:
        logger("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    logger("------------------")

    return cmc[0]


if __name__ == "__main__":
    parser = ArgumentParser(description="Parameters for SDAN")
    parser.add_argument('--mode', type=str, default='train', help='executive mode, train, resume or eval')
    parser.add_argument('--dataset', type=str, default='sysu', help='the used dataset')
    parser.add_argument('--seed', type=int, default=1, help='random seed')

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
    parser.add_argument('--margin', type=float, default=0.3, help='margin of triplet loss')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout possibility')
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
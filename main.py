# -*- coding: utf-8 -*-
import argparse
import os
import time
import cv2
import os.path as osp
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import pprint
from utils import *
from logger import *
from configs.yml_parser import YAMLParser
from easydict import EasyDict

from model.get_model import get_model
from datasets.h5_loader_rssf import *
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--configs', '-c', type=str, default='./configs/spike2flow.yml')
parser.add_argument('--save_dir', '-sd', type=str, default='./outputs')
parser.add_argument('--batch_size', '-bs', type=int, default=6)
parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
parser.add_argument('--num_workers', '-j', type=int, default=12)
parser.add_argument('--start-epoch', '-se', type=int, default=0)
parser.add_argument('--pretrained', '-prt', type=str, default=None)
parser.add_argument('--print_freq', '-pf', type=int, default=None)
parser.add_argument('--vis_path', '-vp', type=str, default='./vis')
parser.add_argument('--model_iters', '-mit', type=int, default=8)
parser.add_argument('--no_warm', '-nw', action='store_true', default=False)
parser.add_argument('--eval', '-e', action='store_true')
parser.add_argument('--save_name', '-sn', type=str, default=None)
parser.add_argument('--warm_iters', '-wi', type=int, default=3000)
parser.add_argument('--eval_vis', '-ev', type=str, default='eval_vis')
parser.add_argument('--crop_len', '-clen', type=int, default=200)
parser.add_argument('--with_valid', '-wv', type=bool, default=True)
parser.add_argument('--decay_interval', '-di', type=int, default=10)
parser.add_argument('--decay_factor', '-df', type=float, default=0.7)
parser.add_argument('--valid_vis_freq', '-vvf', type=float, default=10)
args = parser.parse_args()

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
cfg_parser = YAMLParser(args.configs)
cfg = cfg_parser.config

n_iter = 0


if args.print_freq != None:
    cfg['train']['print_freq'] = args.print_freq
if args.batch_size != None:
    cfg['loader']['batch_size'] = args.batch_size


############################# updata the root ###############################
cfg['data']['spike_path'] = osp.join(cfg['data']['rssf_path'], 'spike')
cfg['data']['dsft_path']  = osp.join(cfg['data']['rssf_path'], 'dsft' )
cfg['data']['flow1_path'] = osp.join(cfg['data']['rssf_path'], 'flow1')
cfg['data']['flow2_path'] = osp.join(cfg['data']['rssf_path'], 'flow2')
cfg['data']['flow3_path'] = osp.join(cfg['data']['rssf_path'], 'flow3')
#############################################################################
    

################################# warm up ###################################
warmup = WarmUp(ed_it=args.warm_iters, st_lr=1e-7, ed_lr=args.learning_rate)
#############################################################################


##########################################################################################################
## Train
def train(cfg, train_loader, model, optimizer, epoch, log, train_writer):
    ######################################################################
    ## Init
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    ######################################################################
    ## Training Loop
    
    for ww, data in enumerate(train_loader, 0):
        if (not args.no_warm) and (n_iter <= args.warm_iters):
            warmup.adjust_lr(optimizer=optimizer, cur_it=n_iter)
        
        st1 = time.time()
        spikes = data['spikes']
        spikes = [spk.cuda() for spk in spikes]
        # spks = torch.cat(spikes, dim=1).cuda()
        spks = torch.cat(spikes, dim=1)
        flow1gt = data['flows'][0].cuda()
        flow2gt = data['flows'][1].cuda()
        flow3gt = data['flows'][2].cuda()
        flowgt = [flow1gt, flow2gt, flow3gt]
        data_time.update(time.time() - end)

        if args.model_iters == None:
            flow = model(spks=spks)
        else:
            flow = model(spks=spks, iters=args.model_iters)
        
        ## compute loss
        loss, loss_deriv_dict = supervised_loss(flow, flowgt)
        
        # record loss
        losses.update(loss.item())
        flow_mean = loss_deriv_dict['flow_mean']
        train_writer.add_scalar('total_loss', loss.item(), n_iter)
        train_writer.add_scalar('flow_mean', flow_mean, n_iter)

        ## compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        n_iter += 1
        if n_iter % cfg['train']['vis_freq'] == 0:
            vis_flow_batch(flow[-1], args.vis_path, suffix='forw_flow', max_batch=16)

        ## output logs
        if ww % cfg['train']['print_freq'] == 0:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            out_str = 'Epoch: [{:d}] [{:d}/{:d}],  Iter: {:d}  '.format(epoch, ww, len(train_loader), n_iter-1)
            out_str += 'Time: {},  Data: {},  Loss: {}, Flow mean {:.4f}, lr {:.7f}'.format(batch_time, data_time, losses, flow_mean, cur_lr)
            log.info(out_str)

        end = time.time()
    
    return


##########################################################################################################
## valid
def validation(cfg, test_datasets, model, log):
    global n_iter
    data_time = AverageMeter()
    AEE1 = AverageMeter()
    AEE2 = AverageMeter()
    AEE3 = AverageMeter()
    F1_1 = AverageMeter()
    F1_2 = AverageMeter()
    F1_3 = AverageMeter()
    model_time = AverageMeter()
    end = time.time()
    epe1_dict = {}
    epe2_dict = {}
    epe3_dict = {}
    f1_1_dict = {}
    f1_2_dict = {}
    f1_3_dict = {}
    len_dict = {}

    # switch to evaluate mode
    model.eval()

    i_set = 0
    for scene, cur_test_set in test_datasets.items():
        i_set += 1
        cur_test_loader = torch.utils.data.DataLoader(
            cur_test_set,
            pin_memory = False,
            drop_last = False,
            batch_size = 1,
            shuffle = False,
            num_workers = args.num_workers)

        cur_aee1 = AverageMeter()
        cur_aee2 = AverageMeter()
        cur_aee3 = AverageMeter()
        cur_f1_1 = AverageMeter()
        cur_f1_2 = AverageMeter()
        cur_f1_3 = AverageMeter()
        cur_model_time = AverageMeter()
        cur_eval_vis_path = osp.join(args.eval_vis, scene)
        make_dir(cur_eval_vis_path)
        for ww, data in enumerate(cur_test_loader, 0):
            spikes = data['spikes']
            spikes = [spk.cuda() for spk in spikes]
            spks = torch.cat(spikes, dim=1)
            flow1gt = data['flows'][0].cuda().permute([0,3,1,2])
            flow2gt = data['flows'][1].cuda().permute([0,3,1,2])
            flow3gt = data['flows'][2].cuda().permute([0,3,1,2])

            data_time.update(time.time() - end)
            with torch.no_grad():
                st = time.time()
                flow = model(spks=spks, iters=args.model_iters)
                mtime = time.time() - st

            if ww % args.valid_vis_freq == 0:
                flow_vis = flow_to_image(flow[-1][0][0].permute([1,2,0]).cpu().numpy())
                cur_vis_path = osp.join(cur_eval_vis_path, '{:03d}.png'.format(ww))
                cv2.imwrite(cur_vis_path, flow_vis)

            # epe
            epe1 = torch.norm(flow[-1][0] - flow1gt, p=2, dim=1).mean()
            epe2 = torch.norm(flow[-1][1] - flow2gt, p=2, dim=1).mean()
            epe3 = torch.norm(flow[-1][2] - flow3gt, p=2, dim=1).mean()
            f1_1 = calculate_error_rate(flow[-1][0], flow1gt)
            f1_2 = calculate_error_rate(flow[-1][1], flow2gt)
            f1_3 = calculate_error_rate(flow[-1][2], flow3gt)

            cur_aee1.update(epe1)
            cur_aee2.update(epe2)
            cur_aee3.update(epe3)
            cur_f1_1.update(f1_1)
            cur_f1_2.update(f1_2)
            cur_f1_3.update(f1_3)

            AEE1.update(epe1)
            AEE2.update(epe2)
            AEE3.update(epe3)
            F1_1.update(f1_1)
            F1_2.update(f1_2)
            F1_3.update(f1_3)

            cur_model_time.update(mtime)
            model_time.update(mtime)
        epe1_dict[scene] = cur_aee1.avg
        epe2_dict[scene] = cur_aee2.avg
        epe3_dict[scene] = cur_aee3.avg
        f1_1_dict[scene] = cur_f1_1.avg
        f1_2_dict[scene] = cur_f1_2.avg
        f1_3_dict[scene] = cur_f1_3.avg

        len_dict[scene] = cur_test_set.__len__()
        log.info('Scene[{:02d}]: {:30s}  EPE1: {:.4f}  EPE2: {:.4f}  EPE3: {:.4f}  AvgTime: {:.4f}'.format(i_set, scene, cur_aee1.avg, cur_aee2.avg, cur_aee3.avg, cur_model_time.avg))
        time.sleep(0.1)
    
    log.info('All EPE1: {:.4f}  All EPE2: {:.4f}  All EPE3: {:.4f}  AvgTime: {:4f}'.format(AEE1.avg, AEE2.avg, AEE3.avg, model_time.avg))
    a_epe1, b_epe1, c_epe1 = get_class_aepe(epe1_dict, len_dict)
    a_epe2, b_epe2, c_epe2 = get_class_aepe(epe2_dict, len_dict)
    a_epe3, b_epe3, c_epe3 = get_class_aepe(epe3_dict, len_dict)
    log.info('EPE1: Class A: {:.4f},  Class B: {:.4f},  Class C: {:.4f}'.format(a_epe1, b_epe1, c_epe1))
    log.info('EPE2: Class A: {:.4f},  Class B: {:.4f},  Class C: {:.4f}'.format(a_epe2, b_epe2, c_epe2))
    log.info('EPE3: Class A: {:.4f},  Class B: {:.4f},  Class C: {:.4f}'.format(a_epe3, b_epe3, c_epe3))
    
    return



if __name__ == '__main__':
    ##########################################################################################################
    # Create save path and logs
    timestamp1 = datetime.datetime.now().strftime('%m-%d')
    timestamp2 = datetime.datetime.now().strftime('%H%M%S')
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    if args.save_name == None:
        save_folder_name = 'b{:d}_{:s}'.format(cfg['loader']['batch_size'], timestamp2)
    else:
        save_folder_name = 'b{:d}_{:s}_{:s}'.format(cfg['loader']['batch_size'], timestamp2, args.save_name)

    save_root = osp.join(args.save_dir, timestamp1)
    save_path = osp.join(save_root, save_folder_name)
    make_dir(args.save_dir)
    make_dir(save_root)
    make_dir(save_path)
    make_dir(args.vis_path)
    make_dir(args.eval_vis)

    _log = init_logger(log_dir=save_path, filename=timestamp2+'.log')
    _log.info('=> will save everything to {:s}'.format(save_path))
    # show configurations
    cfg_str = pprint.pformat(cfg)
    _log.info('=> configurations: \n' + cfg_str)

    train_writer = SummaryWriter(save_path)

    ##########################################################################################################
    ## Create model
    model_dict =  {
            "dropout": 0.0,
            "mixed_precision": True,
            "corr_levels": 3,
            "corr_radius": 3,
            }
    model_dict = EasyDict(model_dict)
    model = get_model(model_dict)
    
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        _log.info('=> using pretrained flow model {:s}'.format(args.pretrained))
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(network_data)
    else:
        network_data = None
        _log.info('=> train flow model from scratch')
        model.init_weights()
        _log.info('=> Flow model params: {:.6f}M'.format(model.num_parameters()/1e6))
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    ##########################################################################################################
    ## Create Optimizer
    cfgopt = cfg['optimizer']
    cfgmdl = cfg['model']
    assert(cfgopt['solver'] in ['Adam', 'SGD'])
    _log.info('=> settings {:s} solver'.format(cfgopt['solver']))
    
    param_groups = [{'params': model.module.parameters(), 'weight_decay': cfgmdl['flow_weight_decay']}]
    if cfgopt['solver'] == 'Adam':
        optimizer = torch.optim.Adam(param_groups, args.learning_rate, betas=(cfgopt['momentum'], cfgopt['beta']))
    elif cfgopt['solver'] == 'SGD':
        optimizer = torch.optim.SGD(param_groups, args.learning_rate, momentum=cfgopt['momentum'])
        
    ##########################################################################################################
    ## Dataset
    train_set = H5Loader_rssf_train(cfg)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        drop_last = True,
        batch_size = cfg['loader']['batch_size'],
        shuffle = True,
        # pin_memory = False,
        # prefetch_factor = 6,
        num_workers = args.num_workers)
    
    if args.eval:
        test_datasets = get_test_datasets(cfg, valid=True, crop_len=args.crop_len)
        validation(cfg=cfg, test_datasets=test_datasets, model=model, log=_log)
    else:
        test_datasets = get_test_datasets(cfg, valid=True, crop_len=args.crop_len)
        epoch = args.start_epoch
        while(True):
            train(
                cfg=cfg,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                log=_log,
                train_writer=train_writer)
            epoch += 1

            if epoch % args.decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * args.decay_factor

            if args.with_valid:
                if epoch % 10 == 0:
                    validation(
                        cfg=cfg, 
                        test_datasets=test_datasets, 
                        model=model, 
                        log=_log)

            # Save Model
            flow_model_save_name = '{:s}_epoch{:03d}.pth'.format(cfg['model']['flow_arch'], epoch)
            torch.save(model.state_dict(), osp.join(save_path, flow_model_save_name))

            if epoch >= cfg['loader']['n_epochs']:
                break
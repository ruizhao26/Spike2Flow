import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.update import BasicUpdateBlock
from model.extractor import BasicEncoder
from model.corr import CorrBlock
from model.utils import bilinear_sampler, coords_grid, upflow8
from model.warp_utils import get_occ_mask_bidirection, flow_warp

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class Spike2Flow(nn.Module):
    def __init__(self, args):
        super(Spike2Flow, self).__init__()
        self.args = args
        
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        if 'dropout' not in self.args:
            self.args.dropout = 0
        if 'mixed_precision' not in self.args:
            self.mixed_precision = True
        if 'corr_levels' not in self.args:
            self.corr_levels = 3
        if 'corr_radius' not in self.args:
            self.corr_radius = 3

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=216, norm_fn='instance', dropout=args.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='instance', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim, input_dim=cdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    ####################################################################################
    ## Tools functions for neural networks
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])
    
    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                
    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)
        coords2 = coords_grid(N, H//8, W//8, device=img.device)
        coords3 = coords_grid(N, H//8, W//8, device=img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1, coords2, coords3

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, spks, iters=8, flow_init=None, upsample=True, test_mode=False, phm=False):
        """ Estimate optical flow between pair of frames """

        if phm:
            spk0 = spks[:, 0:21].contiguous()
            spk1 = spks[:, 10:31].contiguous()
            spk2 = spks[:, 20:41].contiguous()
            spk3 = spks[:, 30:51].contiguous()
        else:
            spk0 = spks[:, 10:31].contiguous()
            spk1 = spks[:, 30:51].contiguous()
            spk2 = spks[:, 50:71].contiguous()
            spk3 = spks[:, 70:91].contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap0, fmap1, fmap2, fmap3 = self.fnet([spk0, spk1, spk2, spk3])        
        
        fmap0 = fmap0.float()
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        fmap3 = fmap3.float()
        
        corr_fn1 = CorrBlock(fmap0, fmap1, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        corr_fn2 = CorrBlock(fmap0, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        corr_fn3 = CorrBlock(fmap0, fmap3, num_levels=self.args.corr_levels, radius=self.args.corr_radius)


        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(spk0)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1, coords2, coords3 = self.initialize_flow(spk0)

        if flow_init is not None:
            coords1 = coords1 + flow_init[0]
            coords2 = coords2 + flow_init[1]
            coords3 = coords3 + flow_init[2]

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            coords2 = coords2.detach()
            coords3 = coords3.detach()
            
            corr1 = corr_fn1(coords1) # index correlation volume
            corr2 = corr_fn2(coords2) # index correlation volume
            corr3 = corr_fn3(coords3) # index correlation volume

            flow1 = coords1 - coords0
            flow2 = coords2 - coords0
            flow3 = coords3 - coords0

            corr_list = [corr1, corr2, corr3]
            flow_list = [flow1, flow2, flow3]
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow_list = self.update_block(net, inp, corr_list, flow_list)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow_list[0]
            coords2 = coords2 + delta_flow_list[1] * 2
            coords3 = coords3 + delta_flow_list[2] * 3

            # upsample predictions
            if up_mask is None:
                flow_up = []
                flow_up.append(upflow8(coords1 - coords0))
                flow_up.append(upflow8(coords2 - coords0))
                flow_up.append(upflow8(coords3 - coords0))
            else:
                flow_up = []
                flow_up.append(self.upsample_flow(coords1 - coords0, up_mask))
                flow_up.append(self.upsample_flow(coords2 - coords0, up_mask))
                flow_up.append(self.upsample_flow(coords3 - coords0, up_mask))
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords3 - coords0, flow_up
            
        return flow_predictions

    
    
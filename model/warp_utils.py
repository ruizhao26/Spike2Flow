# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_corresponding_map(data):
    """

    :param data: unnormalized coordinates Bx2xHxW
    :return: Bx1xHxW
    """
    B, _, H, W = data.size()

    # x = data[:, 0, :, :].view(B, -1).clamp(0, W - 1)  # BxN (N=H*W)
    # y = data[:, 1, :, :].view(B, -1).clamp(0, H - 1)

    x = data[:, 0, :, :].view(B, -1)  # BxN (N=H*W)
    y = data[:, 1, :, :].view(B, -1)

    # invalid = (x < 0) | (x > W - 1) | (y < 0) | (y > H - 1)   # BxN
    # invalid = invalid.repeat([1, 4])

    x1 = torch.floor(x)
    x_floor = x1.clamp(0, W - 1)
    y1 = torch.floor(y)
    y_floor = y1.clamp(0, H - 1)
    x0 = x1 + 1
    x_ceil = x0.clamp(0, W - 1)
    y0 = y1 + 1
    y_ceil = y0.clamp(0, H - 1)

    x_ceil_out = x0 != x_ceil
    y_ceil_out = y0 != y_ceil
    x_floor_out = x1 != x_floor
    y_floor_out = y1 != y_floor
    invalid = torch.cat([x_ceil_out | y_ceil_out,
                         x_ceil_out | y_floor_out,
                         x_floor_out | y_ceil_out,
                         x_floor_out | y_floor_out], dim=1)

    # encode coordinates, since the scatter function can only index along one axis
    corresponding_map = torch.zeros(B, H * W).type_as(data)
    indices = torch.cat([x_ceil + y_ceil * W,
                         x_ceil + y_floor * W,
                         x_floor + y_ceil * W,
                         x_floor + y_floor * W], 1).long()  # BxN   (N=4*H*W)
    values = torch.cat([(1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_ceil)) * (1 - torch.abs(y - y_floor)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_ceil)),
                        (1 - torch.abs(x - x_floor)) * (1 - torch.abs(y - y_floor))],
                       1)
    # values = torch.ones_like(values)

    values[invalid] = 0

    corresponding_map.scatter_add_(1, indices, values)
    # decode coordinates
    corresponding_map = corresponding_map.view(B, H, W)

    return corresponding_map.unsqueeze(1)


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


# occ_mask = 0 : The pixel is not occluded
# occ_mask = 1 : The pixel is occluded
def get_occ_mask_bidirection(flow12, flow21, scale=0.01, bias=0.5):
    flow21_warped = flow_warp(flow21, flow12, pad='zeros')
    flow12_diff = flow12 + flow21_warped
    mag = (flow12 * flow12).sum(1, keepdim=True) + \
          (flow21_warped * flow21_warped).sum(1, keepdim=True)
    occ_thresh = scale * mag + bias
    occ = (flow12_diff * flow12_diff).sum(1, keepdim=True) > occ_thresh
    return occ.float()


def get_occ_mask_backward(flow21, th=0.2):
    B, _, H, W = flow21.size()
    base_grid = mesh_grid(B, H, W).type_as(flow21)  # B2HW

    corr_map = get_corresponding_map(base_grid + flow21)  # BHW
    occu_mask = corr_map.clamp(min=0., max=1.) < th
    return occu_mask.float()


# boundary dilated warping
class boundary_dilated_warp():

    @classmethod
    def get_grid(cls, batch_size, H, W, start):
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(batch_size, 1, 1, 1)
        ones = torch.ones_like(xx)
        grid = torch.cat((xx, yy, ones), 1).float()
        if torch.cuda.is_available():
            grid = grid.cuda()
        # print("grid",grid.shape)
        # print("start", start)
        grid[:, :2, :, :] = grid[:, :2, :, :] + start  # 加上patch在原图内的偏移量

        return grid

    @classmethod
    def transformer(cls, I, vgrid, train=True):
        # I: Img, shape: batch_size, 1, full_h, full_w
        # vgrid: vgrid, target->source, shape: batch_size, 2, patch_h, patch_w
        # outsize: (patch_h, patch_w)

        def _repeat(x, n_repeats):

            rep = torch.ones([n_repeats, ]).unsqueeze(0)
            rep = rep.int()
            x = x.int()

            x = torch.matmul(x.reshape([-1, 1]), rep)
            return x.reshape([-1])

        def _interpolate(im, x, y, out_size, scale_h):
            # x: x_grid_flat
            # y: y_grid_flat
            # out_size: same as im.size
            # scale_h: True if normalized
            # constants
            num_batch, num_channels, height, width = im.size()

            out_height, out_width = out_size[0], out_size[1]
            # zero = torch.zeros_like([],dtype='int32')
            zero = 0
            max_y = height - 1
            max_x = width - 1
            if scale_h:
                # scale indices from [-1, 1] to [0, width or height]
                # print('--Inter- scale_h:', scale_h)
                x = (x + 1.0) * (height) / 2.0
                y = (y + 1.0) * (width) / 2.0

            # do sampling
            x0 = torch.floor(x).int()
            x1 = x0 + 1
            y0 = torch.floor(y).int()
            y1 = y0 + 1

            x0 = torch.clamp(x0, zero, max_x)  # same as np.clip
            x1 = torch.clamp(x1, zero, max_x)
            y0 = torch.clamp(y0, zero, max_y)
            y1 = torch.clamp(y1, zero, max_y)

            dim1 = torch.from_numpy(np.array(width * height))
            dim2 = torch.from_numpy(np.array(width))

            base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)  # 其实就是单纯标出batch中每个图的下标位置
            # base = torch.arange(0,num_batch) * dim1
            # base = base.reshape(-1, 1).repeat(1, out_height * out_width).reshape(-1).int()
            # 区别？expand不对数据进行拷贝 .reshape(-1,1).expand(-1,out_height * out_width).reshape(-1)
            if torch.cuda.is_available():
                dim2 = dim2.cuda()
                dim1 = dim1.cuda()
                y0 = y0.cuda()
                y1 = y1.cuda()
                x0 = x0.cuda()
                x1 = x1.cuda()
                base = base.cuda()
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im = im.permute(0, 2, 3, 1)
            im_flat = im.reshape([-1, num_channels]).float()

            idx_a = idx_a.unsqueeze(-1).long()
            idx_a = idx_a.expand(out_height * out_width * num_batch, num_channels)
            Ia = torch.gather(im_flat, 0, idx_a)

            idx_b = idx_b.unsqueeze(-1).long()
            idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
            Ib = torch.gather(im_flat, 0, idx_b)

            idx_c = idx_c.unsqueeze(-1).long()
            idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
            Ic = torch.gather(im_flat, 0, idx_c)

            idx_d = idx_d.unsqueeze(-1).long()
            idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
            Id = torch.gather(im_flat, 0, idx_d)

            # and finally calculate interpolated values
            x0_f = x0.float()
            x1_f = x1.float()
            y0_f = y0.float()
            y1_f = y1.float()

            wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
            wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
            wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
            wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
            output = wa * Ia + wb * Ib + wc * Ic + wd * Id

            return output

        def _transform(I, vgrid, scale_h):

            C_img = I.shape[1]
            B, C, H, W = vgrid.size()

            x_s_flat = vgrid[:, 0, ...].reshape([-1])
            y_s_flat = vgrid[:, 1, ...].reshape([-1])
            out_size = vgrid.shape[2:]
            input_transformed = _interpolate(I, x_s_flat, y_s_flat, out_size, scale_h)

            output = input_transformed.reshape([B, H, W, C_img])
            return output

        # scale_h = True
        output = _transform(I, vgrid, scale_h=False)
        if train:
            output = output.permute(0, 3, 1, 2)
        return output

    @classmethod
    def warp_im(cls, I_nchw, flow_nchw, start_n211):
        batch_size, _, img_h, img_w = I_nchw.size()
        _, _, patch_size_h, patch_size_w = flow_nchw.size()
        patch_indices = cls.get_grid(batch_size, patch_size_h, patch_size_w, start_n211)
        vgrid = patch_indices[:, :2, ...]
        # grid_warp = vgrid - flow_nchw
        grid_warp = vgrid + flow_nchw
        pred_I2 = cls.transformer(I_nchw, grid_warp)
        return pred_I2

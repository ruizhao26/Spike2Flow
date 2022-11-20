import numpy as np
import os
import json
import argparse
from ds_utils import *
import os.path as osp
import cv2
import time
import torch
from tqdm import *

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


parser = argparse.ArgumentParser( )
parser.add_argument('-rr', '-rssf_root', type=str, default="/home/data/rzhao/rssf")
parser.add_argument('-ds', '--dataset', type=str, default='rssf')
parser.add_argument('-sc', '--img_size_config', type=str, default='img_size.json')
parser.add_argument('-fl', '--flipud', action='store_true', help='Flip the raw spike')
parser.add_argument("--scenario-ids", '-sid', nargs='+', default=None)
parser.add_argument("--device", default='cuda')
args = parser.parse_args()

args.data_root = osp.join(args.rssf_root, 'spike')
args.out_root = osp.join(args.rssf_root, 'dsft')

SEARCH_WINDOW = 40
def Get_dsft_torch(sp_mat, key_id, search_window=SEARCH_WINDOW):
    # key_id count as the FORMMER GROUP
    # FORMMER GROUP k-(search_window - 1)
    # LATTER GROUP k+search_window
    c, h, w = sp_mat.shape
    formmer_index = torch.zeros([h, w]).to(args.device)
    latter_index = torch.zeros([h, w]).to(args.device)
    
    start_t = max(key_id - search_window + 1, 1)
    end_t = min(key_id + search_window, c)

    for ii in range(key_id, start_t-1, -1):
        formmer_index += ii * sp_mat[ii, :, :] * (1 - torch.sign(formmer_index).to(args.device))

    for ii in range(key_id+1, end_t+1):
        latter_index += ii * sp_mat[ii, :, :] * (1 - torch.sign(latter_index).to(args.device))


    dsft = latter_index - formmer_index
    dsft[dsft == 0] = 2*search_window
    dsft[latter_index == 0] = 2*search_window
    dsft[formmer_index == 0] = 2*search_window
    return dsft


if __name__ == '__main__':
    img_size_dict = json.load(open(args.img_size_config))

    spk_file_path = osp.join(args.data_root)
    out_interp_path = osp.join(args.out_root)
    make_dir(args.out_root)
    make_dir(out_interp_path)
    scenario_list = sorted(os.listdir(spk_file_path))

    # for scenario in scenario_list:
    for kk in range(len(scenario_list)):
        if str(kk) not in args.scenario_ids:
            continue
        scenario = scenario_list[kk]
        cur_scenario_path = osp.join(spk_file_path, scenario)
        out_scenario_path = osp.join(out_interp_path, scenario)
        make_dir(out_scenario_path)
        cur_dat_list = sorted(os.listdir(cur_scenario_path))

        print('converting to h5 for scenario {:s}:'.format(scenario))
        for ii in tqdm(range(len(cur_dat_list))):
            if(ii<2 or ii>len(cur_dat_list)-3):
                continue
            h5file_name = cur_dat_list[ii][:-4] + '.h5'
            cur_h5_path = osp.join(out_scenario_path, h5file_name)
            for jj in range(-2, -2+5, 1):
                cur_dat_path = osp.join(cur_scenario_path, cur_dat_list[ii+jj])
                cur_spmat = dat_to_spmat(cur_dat_path, size=img_size_dict[args.dataset][scenario])
                if jj == -2:
                    spmat = cur_spmat
                else:
                    spmat = np.concatenate([spmat, cur_spmat], axis=0)

            dsft_mat = np.zeros([20]+img_size_dict[args.dataset][scenario]).astype(np.uint8)            
            spmat = torch.from_numpy(spmat).to(args.device)
            for jj in range(40, 60):
                dsft_mat[jj-40, :, :] = Get_dsft_torch(spmat, jj).cpu().numpy().astype(np.uint8)

            save_to_h5(dsft_mat, cur_h5_path)
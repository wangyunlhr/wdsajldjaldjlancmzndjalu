import torch
from torch.utils.data import Dataset
from lidiff.utils.pcd_preprocess import point_set_to_coord_feats, aggregate_pcds, load_poses
from lidiff.utils.pcd_transforms import *
from lidiff.utils.data_map import learning_map
from lidiff.utils.collations import point_preprocess
from natsort import natsorted
import os
import numpy as np
import yaml

import warnings
import pickle
import torch

import h5py
from tqdm import tqdm

warnings.filterwarnings('ignore')

#################################################
################## Data loader ##################
#################################################

class HDF5Dataset(Dataset):
    def __init__(self, directory, split, max_range=50, eval = False, leaderboard_version=1):
        super().__init__()
        super(HDF5Dataset, self).__init__()
        self.directory = directory + '/sensor/' + split
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)[:32]

        self.eval_index = False

        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if leaderboard_version == 2:
                print("Using index to leaderboard version 2!!")
                eval_index_file = os.path.join(BASE_DIR, 'assets/docs/index_eval_v2.pkl')
            if not os.path.exists(eval_index_file):
                raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval
            with open(eval_index_file, 'rb') as f:
                self.eval_data_index = pickle.load(f)
        
        self.max_range = max_range


    def transforms(self, points):
        points = np.expand_dims(points, axis=0)
        points[:,:,:3] = rotate_point_cloud(points[:,:,:3])
        points[:,:,:3] = rotate_perturbation_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_scale_point_cloud(points[:,:,:3])
        points[:,:,:3] = random_flip_point_cloud(points[:,:,:3])

        return np.squeeze(points, axis=0)

    # def __getitem__(self, index):

    #     seq_num = self.points_datapath[index].split('/')[-3]
    #     fname = self.points_datapath[index].split('/')[-1].split('.')[0]

    #     p_part = np.fromfile(self.points_datapath[index], dtype=np.float32)
    #     p_part = p_part.reshape((-1,4))[:,:3]
        
    #     if self.split != 'test':
    #         label_file = self.points_datapath[index].replace('velodyne', 'labels').replace('.bin', '.label')
    #         l_set = np.fromfile(label_file, dtype=np.uint32)
    #         l_set = l_set.reshape((-1))
    #         l_set = l_set & 0xFFFF
    #         static_idx = (l_set < 252) & (l_set > 1)
    #         p_part = p_part[static_idx]
    #     dist_part = np.sum(p_part**2, -1)**.5
    #     p_part = p_part[(dist_part < self.max_range) & (dist_part > 3.5)]
    #     p_part = p_part[p_part[:,2] > -4.]
    #     pose = self.seq_poses[index] # 当前帧的lidar坐标系 to 世界坐标系

    #     p_map = self.cache_maps[seq_num]

    #     if self.split != 'test':
    #         trans = pose[:-1,-1]
    #         dist_full = np.sum((p_map - trans)**2, -1)**.5
    #         p_full = p_map[dist_full < self.max_range]
    #         p_full = np.concatenate((p_full, np.ones((len(p_full),1))), axis=-1)
    #         p_full = (p_full @ np.linalg.inv(pose).T)[:,:3]
    #         p_full = p_full[p_full[:,2] > -4.]
    #     else:
    #         p_full = p_part

    #     if self.split == 'train':
    #         p_concat = np.concatenate((p_full, p_part), axis=0)
    #         p_concat = self.transforms(p_concat)

    #         p_full = p_concat[:-len(p_part)]
    #         p_part = p_concat[-len(p_part):]

    #     # patial pcd has 1/10 of the complete pcd size
    #     n_part = int(self.num_points / 10.)

    #     return point_set_to_sparse(
    #         p_full,
    #         p_part,
    #         self.num_points,
    #         n_part,
    #         self.resolution, # 0.05
    #         self.points_datapath[index],
    #         p_mean=self.data_stats['mean'],
    #         p_std=self.data_stats['std'],
    #     )

    def __getitem__(self, index_):
        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp])
        else:
            scene_id, timestamp = self.data_index[index_]


        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            pc0 = torch.tensor(f[key]['pcl_0'][:][:,:3])
            pc1 = torch.tensor(f[key]['pcl_1'][:][:,:3])
            pc2 = torch.tensor(f[key]['pcl_2'][:][:,:3])
            pc3 = torch.tensor(f[key]['pcl_3'][:][:,:3])
            pc4 = torch.tensor(f[key]['pcl_4'][:][:,:3])

            flow_0 = torch.tensor(f[key]['flow_0'][:])
            flow_1 = torch.tensor(f[key]['flow_1'][:])
            flow_2 = torch.tensor(f[key]['flow_2'][:])
            flow_3 = torch.tensor(f[key]['flow_3'][:])

            valid_0 = torch.tensor(f[key]['valid_0'][:])
            valid_1 = torch.tensor(f[key]['valid_1'][:])
            valid_2 = torch.tensor(f[key]['valid_2'][:])
            valid_3 = torch.tensor(f[key]['valid_3'][:])

            classes_0 = torch.tensor(f[key]['classes_0'][:])
            classes_1 = torch.tensor(f[key]['classes_1'][:])
            classes_2 = torch.tensor(f[key]['classes_2'][:])
            classes_3 = torch.tensor(f[key]['classes_3'][:])

            ground_mask_0 = torch.tensor(f[key]['ground_mask_0'][:])
            ground_mask_1 = torch.tensor(f[key]['ground_mask_1'][:])
            ground_mask_2 = torch.tensor(f[key]['ground_mask_2'][:])
            ground_mask_3 = torch.tensor(f[key]['ground_mask_3'][:])
            ground_mask_4 = torch.tensor(f[key]['ground_mask_4'][:])

            p_0_3 = torch.cat([pc0, pc1, pc2, pc3], dim=0) # dense point cloud
            p_0_3_gm_mask = torch.cat([ground_mask_0, ground_mask_1, ground_mask_2, ground_mask_3], dim=0)
            p_4 = pc4
            p_4_gm_mask = ground_mask_4

            flow_0_3 = torch.cat([flow_0, flow_1, flow_2, flow_3], dim=0)
            valid_0_3 = torch.cat([valid_0, valid_1, valid_2, valid_3], dim=0)
            classes_0_3 = torch.cat([classes_0, classes_1, classes_2, classes_3], dim=0)

            dist_0_3 = (torch.norm(p_0_3, dim=1) < self.max_range)
            dist_4 = (torch.norm(p_4, dim=1) < self.max_range)
            filename = f'{scene_id}_{timestamp}'
            res_dict = {
                'p_0_3': p_0_3,
                'p_0_3_gm_mask': p_0_3_gm_mask,
                'p_4': p_4,
                'p_4_gm_mask': p_4_gm_mask,
                'flow_0_3': flow_0_3,
                'valid_0_3': valid_0_3,
                'classes_0_3': classes_0_3,
                'filename': filename,
                'dist_0_3': dist_0_3,
                'dist_4': dist_4,
            }     

            if self.eval_index:
                # looks like v2 not follow the same rule as v1 with eval_mask provided
                eval_mask = torch.tensor(f[key]['eval_mask'][:]) if 'eval_mask' in f[key] else torch.ones_like(pc0[:, 0], dtype=torch.bool)
                res_dict['eval_mask'] = eval_mask

        return point_preprocess(res_dict)

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)

##################################################################################################

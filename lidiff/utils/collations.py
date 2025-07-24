import numpy as np
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

def feats_to_coord(p_feats, resolution, mean, std):
    p_feats = p_feats.reshape(mean.shape[0],-1,3)
    p_coord = torch.round(p_feats / resolution)

    return p_coord.reshape(-1,3)

def normalize_pcd(points, mean, std):
    return (points - mean[:,None,:]) / std[:,None,:] if len(mean.shape) == 2 else (points - mean) / std

def unormalize_pcd(points, mean, std):
    return (points * std[:,None,:]) + mean[:,None,:] if len(mean.shape) == 2 else (points * std) + mean

def point_set_to_sparse_refine(p_full, p_part, n_full, n_part, resolution, filename):
    concat_full = np.ceil(n_full / p_full.shape[0])
    concat_part = np.ceil(n_part / p_part.shape[0])

    #if mode == 'diffusion':
    #p_full = p_full[torch.randperm(p_full.shape[0])]
    #p_part = p_part[torch.randperm(p_part.shape[0])]
    #elif mode == 'refine':
    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = torch.tensor(p_full.repeat(concat_full, 0)[:n_full])   

    p_part = p_part[torch.randperm(p_part.shape[0])]
    p_part = torch.tensor(p_part.repeat(concat_part, 0)[:n_part])

    #p_feats = ME.utils.batched_coordinates([p_feats], dtype=torch.float32)[:2000]
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean, p_std = p_full.mean(axis=0), p_full.std(axis=0)

    return [p_full, p_mean, p_std, p_part, filename]

def point_set_to_sparse(p_full, p_part, n_full, n_part, resolution, filename, p_mean=None, p_std=None):
    concat_part = np.ceil(n_part / p_part.shape[0]) 
    p_part = p_part.repeat(concat_part, 0)
    pcd_part = o3d.geometry.PointCloud()
    pcd_part.points = o3d.utility.Vector3dVector(p_part)
    viewpoint_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_part, voxel_size=10.)
    pcd_part = pcd_part.farthest_point_down_sample(n_part)
    p_part = torch.tensor(np.array(pcd_part.points))
    
    in_viewpoint = viewpoint_grid.check_if_included(o3d.utility.Vector3dVector(p_full))
    p_full = p_full[in_viewpoint] 
    concat_full = np.ceil(n_full / p_full.shape[0])

    p_full = p_full[torch.randperm(p_full.shape[0])]
    p_full = p_full.repeat(concat_full, 0)[:n_full]

    p_full = torch.tensor(p_full)
    
    # after creating the voxel coordinates we normalize the floating coordinates towards mean=0 and std=1
    p_mean = p_full.mean(axis=0) if p_mean is None else p_mean
    p_std = p_full.std(axis=0) if p_std is None else p_std

    return [p_full, p_mean, p_std, p_part, filename]


def point_preprocess(res_dict):

    mask_0_3 = (~res_dict['p_0_3_gm_mask'] & res_dict['dist_0_3'])
    mask_4 = (~res_dict['p_4_gm_mask'] & res_dict['dist_4'])

    p_0_3 = res_dict['p_0_3'][mask_0_3]
    p_4 = res_dict['p_4'][mask_4]
    flow_0_3 = res_dict['flow_0_3'][mask_0_3]
    valid_0_3 = res_dict['valid_0_3'][mask_0_3]
    classes_0_3 = res_dict['classes_0_3'][mask_0_3]

    max_points = 10000
    if p_0_3.shape[0] > max_points:
        num_points = p_0_3.shape[0]
        indices = torch.randperm(num_points)[:max_points]  # 随机选 keep_num 个索引
        p_0_3 = p_0_3[indices]
        flow_0_3 = flow_0_3[indices]        
        valid_0_3 = valid_0_3[indices]
        classes_0_3 = classes_0_3[indices]

    p_full = p_0_3 + flow_0_3
    p_part = p_4
    filename = res_dict['filename']
    return [p_full, torch.ones(1), torch.ones(1), p_part, filename, flow_0_3, valid_0_3, classes_0_3]

def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(p_coord, dtype=torch.float32)
    p_feats = torch.vstack(p_feats).float()

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(p_label, device=torch.device('cpu')).numpy()
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class SparseSegmentCollation:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))

        return {'pcd_full': torch.stack(batch[0]).float(),
            'mean': torch.stack(batch[1]).float(),
            'std': torch.stack(batch[2]).float(),
            'pcd_part' if self.mode == 'diffusion' else 'pcd_noise': torch.stack(batch[3]).float(),
            'filename': batch[4],
        }



class SparseSegmentCollation_av2:
    def __init__(self, mode='diffusion'):
        self.mode = mode
        return

    def __call__(self, data):
        # "transpose" the  batch(pt, ptn) to batch(pt), batch(ptn)
        batch = list(zip(*data))
        #填充，保证shape一致
        # return {'pcd_full': torch.stack(batch[0]).float(),
        #     'mean': torch.stack(batch[1]).float(),
        #     'std': torch.stack(batch[2]).float(),
        #     'pcd_part': torch.stack(batch[3]).float(),
        #     'filename': batch[4],
        #     'flow': torch.stack(batch[5]).float(),
        #     'valid': torch.stack(batch[6]).float(),
        #     'classes': torch.stack(batch[7]).float(),
        # }

        return {'pcd_full': torch.nn.utils.rnn.pad_sequence(batch[0], batch_first=True, padding_value=torch.nan),
            'mean': torch.stack(batch[1]).float(),
            'std': torch.stack(batch[2]).float(),
            'pcd_part': torch.nn.utils.rnn.pad_sequence(batch[3], batch_first=True, padding_value=torch.nan),
            'filename': batch[4],
            'flow': torch.nn.utils.rnn.pad_sequence(batch[5], batch_first=True),
            'valid': torch.nn.utils.rnn.pad_sequence(batch[6], batch_first=True),
            'classes': torch.nn.utils.rnn.pad_sequence(batch[7], batch_first=True),
        }

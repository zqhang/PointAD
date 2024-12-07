import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec_pc_3d_rgb':
        obj_list = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",]
  
    elif dataset_name == 'eye_pc_3d_rgb':
        obj_list = [
       'CandyCane',
        'ChocolateCookie',
        'ChocolatePraline',
        'Confetto',
        'GummyBear',
        'HazelnutTruffle',
        'LicoriceSandwich',
        'Lollipop',
        'Marshmallow',
        'PeppermintCandy']
    elif dataset_name == 'real_pc_3d_rgb':
        obj_list = ['airplane','car','candybar','chicken',
                 'diamond','duck','fish','gemstone',
                 'seahorse','shell','starfish','toffees']


    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id


def generate_is_seen_for_point_in_each_view(d2_3d_cor_list, non_zero_index_arr):
    # nv, l, 3
    d2_3d_cor = torch.stack(d2_3d_cor_list)
    # nv, l
    is_seen = d2_3d_cor[..., 2]
    nonzero_index = np.nonzero(np.asarray(non_zero_index_arr).reshape(-1,))[0]
    max_nonzero_index, min_nonzero_index = nonzero_index[-1], nonzero_index[0]
    mask = np.asarray(is_seen.bool() | (~non_zero_index_arr.permute(1, 0).bool()))
    # Few points have no projections in any views, we fill the d2_3d_cor of its neighbors as its projection to avoid the nan value when back-projecting 2d presentations to 3d
    is_seen = np.nonzero(np.all(mask == 0, axis = 0))[0]
    
    if len(is_seen):
        for i in is_seen:
            success_find = 0
            up_select_idx = 0
            down_select_idx = 0
            for j in range(i+1, max_nonzero_index + 1):
                up_select_idx = j
                if np.any(d2_3d_cor[..., j, 2].numpy()):
                    success_find = 1
                    dis = np.abs(up_select_idx - i)
                    select_index = up_select_idx
                    break
            if not success_find:
                for j in range(min_nonzero_index, i):
                    up_select_idx = j
                    if np.any(d2_3d_cor[..., j, 2].numpy()):
                        success_find = 1
                        dis = np.abs(max_nonzero_index - i + up_select_idx - min_nonzero_index)
                        select_index = up_select_idx
                        break
            if not success_find:
                raise NotImplementedError("bug")
            for j in reversed(range(min_nonzero_index, i)):
                down_select_idx = j
                if dis <= np.abs(down_select_idx - i):
                    select_index = up_select_idx
                    break;
                else:
                    if np.any(d2_3d_cor[..., j, 2].numpy()):
                        success_find = 1
                        select_index = down_select_idx
                        break
            if not success_find:
                for j in reversed(range(i+1, max_nonzero_index + 1)):
                    down_select_idx = j
                    if dis <= np.abs(i - min_nonzero_index + max_nonzero_index - down_select_idx):
                        select_index = up_select_idx
                        break;
                    else:
                        if np.any(d2_3d_cor[..., j, 2].numpy()):
                            success_find = 1
                            select_index = down_select_idx
                            break
            if not success_find:
                raise NotImplementedError("bug")
            d2_3d_cor[..., i, 2]=d2_3d_cor[..., select_index, 2]
    is_seen = d2_3d_cor[..., 2]
    mask = np.asarray(is_seen.bool() | (~non_zero_index_arr.permute(1, 0).bool()))
    is_seen = np.nonzero(np.all(mask == 0, axis = 0))[0]
    return is_seen, d2_3d_cor


import open3d
class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, target_transform_pc, dataset_name, train_dataset_name = None, point_size = 336, is_all = False, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.target_transform_pc = target_transform_pc
        self.point_size = point_size
        self.dataset_name = dataset_name
        self.data_all = []
        if is_all:
            meta_info = json.load(open(f'{self.root}/all_meta.json', 'r'))
        else:
            if mode == 'test':
                meta_info = json.load(open(f'{self.root}/{train_dataset_name}_meta.json', 'r'))
            else:
                meta_info = json.load(open(f'{self.root}/{obj_name}_meta.json', 'r'))
            # meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        # meta_info = json.load(open(f'/remote-home/iot_zhouqihang/data/mvtec_3d_mv/mvtec_3d_9_views/meta_test.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]

        if self.dataset_name == 'mvtec_pc_3d_rgb' or self.dataset_name == 'eye_pc_3d_rgb':
            img_path, mask_path, cls_name, specie_name, anomaly = data['d2_img_path'], data['d2_mask_path'], data['cls_name'], \
                                                                data['specie_name'], data['anomaly']
            d2_render_img_path, d2_render_gt_path, d2_corrdinate_path = data['d2_render_img_path'], data['d2_render_gt_path'], data['d2_corrdinate']
        
        elif self.dataset_name == 'real_pc_3d_rgb':
            mask_path, cls_name, specie_name, anomaly =  data['d2_mask_path'], data['cls_name'], \
                                                                data['specie_name'], data['anomaly']
            d2_render_img_path, d2_render_gt_path, d2_corrdinate_path = data['d2_render_img_path'], data['d2_render_gt_path'], data['d2_corrdinate']

        # load 2d rendering images
        d2_render_img_path_list = []
        for filename in sorted(os.listdir(d2_render_img_path)):
            img = Image.open(os.path.join(d2_render_img_path, filename)).convert("RGB")
            img = self.transform(img) if self.transform is not None else img
            d2_render_img_path_list.append(img)
        d2_render_img = torch.stack(d2_render_img_path_list)

        # load 2d rendering groundtruth
        d2_render_gt_path_list = []
        rendering_anomaly_list = []
        for filename in sorted(os.listdir(d2_render_gt_path)):
            img_mask = Image.open((os.path.join(d2_render_gt_path, filename))).convert('L')
            img_mask = self.target_transform(img_mask)
            img_mask[img_mask>0.5] = 1.0
            img_mask[img_mask<=0.5] = 0.0
            rendering_anomaly = 0.0 if torch.all(img_mask == 0) else 1.0
            d2_render_gt_path_list.append(img_mask)
            rendering_anomaly_list.append(rendering_anomaly)
        d2_render_gt = torch.stack(d2_render_gt_path_list)
        rendering_anomaly = torch.tensor(rendering_anomaly_list)

        # load the correspondence between points and pixels in each view
        d2_3d_cor_list = []
        non_zero_index_list = []
        # for organized point ckoud
        if self.dataset_name == 'mvtec_pc_3d_rgb' or self.dataset_name == 'eye_pc_3d_rgb':
            for idx, filename in enumerate(sorted(os.listdir(d2_corrdinate_path))):
                if idx == 0:
                    template_non_zero_index = torch.zeros(self.point_size * self.point_size, dtype = torch.long)
                    non_zero_index = np.load((os.path.join(d2_corrdinate_path, filename)))
                    non_zero_index = torch.from_numpy(non_zero_index)
                    template_non_zero_index[non_zero_index] = 1
                    non_zero_index_arr = template_non_zero_index.reshape(-1, 1)
                else:
                    template_d2_corrdinate = torch.zeros(self.point_size * self.point_size, 3, dtype = torch.long)
                    d2_corrdinate = np.load((os.path.join(d2_corrdinate_path, filename)))

                    d2_corrdinate = torch.from_numpy(d2_corrdinate).long()
                    template_d2_corrdinate[non_zero_index] = d2_corrdinate
                    d2_3d_cor_list.append(template_d2_corrdinate)
        # for unorganized point cloud
        elif self.dataset_name == 'real_pc_3d_rgb':
            for idx, filename in enumerate(sorted(os.listdir(d2_corrdinate_path))):
                if idx == 0:
                    template_non_zero_index = torch.ones(self.point_size * self.point_size, dtype = torch.long)
                    non_zero_index_arr = template_non_zero_index.reshape(-1, 1)
                d2_corrdinate = np.load((os.path.join(d2_corrdinate_path, filename)))
                d2_corrdinate = torch.from_numpy(d2_corrdinate).long()
                template_d2_corrdinate = d2_corrdinate
                d2_3d_cor_list.append(template_d2_corrdinate)

        # remove the hidden points in each view
        is_seen, d2_3d_cor = generate_is_seen_for_point_in_each_view(d2_3d_cor_list, non_zero_index_arr)
        
        if self.dataset_name == 'mvtec_pc_3d_rgb' or self.dataset_name == 'eye_pc_3d_rgb':
            img = Image.open(os.path.join(self.root, img_path))
            if anomaly == 0:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                if os.path.isdir(os.path.join(self.root, mask_path)):
                    img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
                else:
                    img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                    img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            # transforms
            img = self.transform(img) if self.transform is not None else img
            img_mask = self.target_transform(   
                img_mask) if self.target_transform is not None and img_mask is not None else img_mask
            img_mask = [] if img_mask is None else img_mask
            return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'd2_render_img': d2_render_img, 'd2_render_anomaly': rendering_anomaly, 'd2_render_gt': d2_render_gt, 'd2_3d_cor': d2_3d_cor,
                    'img_path': os.path.join(self.root, img_path), "cls_id":self.class_name_map_class_id[cls_name], "d2_render_img_path": d2_render_img_path, "non_zero_index": non_zero_index_arr, "index":torch.LongTensor([index])}
        
        elif self.dataset_name == 'real_pc_3d_rgb':
            pcd = open3d.io.read_point_cloud(os.path.join(self.root, mask_path))
            img_mask = np.array(pcd.colors)
            img_mask[img_mask>0.5] = 1.0
            img_mask[img_mask<0.5] = 0.0
            img_mask = np.all(img_mask, axis = 1).astype(int)
            return {'img': '', 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly, 'd2_render_img': d2_render_img, 'd2_render_anomaly': rendering_anomaly, 'd2_render_gt': d2_render_gt, 'd2_3d_cor': d2_3d_cor,
                    'img_path': '',  "cls_id":self.class_name_map_class_id[cls_name], "d2_render_img_path": d2_render_img_path, "non_zero_index": non_zero_index_arr}
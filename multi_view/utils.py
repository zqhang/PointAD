import numpy as np
import tifffile as tiff
import torch


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=False,mode="nearest"):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).float()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode=mode)
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze(dim=0).permute(1, 2, 0).numpy()


def organized_pc_to_depth_map(organized_pc):
    return organized_pc[:, :, 2]

def matrix_z(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle, 0, 0], [sin_angle, cos_angle, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def matrix_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1, 0, 0, 0], 
                     [0, cos_angle, -sin_angle, 0], 
                     [0, sin_angle, cos_angle, 0], 
                     [0, 0, 0, 1]])

def matrix_y(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, 0, sin_angle, 0],
                     [0, 1, 0, 0],
                     [-sin_angle, 0, cos_angle, 0],
                     [0, 0, 0, 1]])

def compute_visibility_3d_to_2d(points_3d, correspondence, img_shape=(500, 500)):
    # Initialize visibility array with zeros
    visibility = np.zeros(len(points_3d), dtype=np.int32)
    
    # Initialize depth map with -inf values
    depth_map = np.full(img_shape, -np.inf)
    
    max_depth_indices = -np.ones(img_shape, dtype=np.int32)
    
    for i, (x, y) in enumerate(correspondence):
        z = points_3d[i, 2]
        if z > depth_map[y, x]:
            depth_map[y, x] = z
            max_depth_indices[y, x] = i
    
    for i, (x, y) in enumerate(correspondence):
        if max_depth_indices[y, x] == i:
            visibility[i] = 1
    return visibility
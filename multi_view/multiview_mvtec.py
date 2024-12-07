from .utils import *
import numpy as np

from PIL import Image
from torchvision import transforms
import open3d as o3d
import copy
import cv2
import torch.nn.functional as F
import os
import ipdb
    
def render_and_save_point_cloud(o3d_pc_r, o3d_pc_gt_r, view_idx, cor_save_path, view_save_path, gt_save_path, point_size):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=1024, visible=False)
    vis.add_geometry(o3d_pc_r)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1, 1, 1])
    render_option.light_on = True
    render_option.point_size = point_size
    render_option.mesh_show_back_face = True
    
    control = vis.get_view_control()
    param = control.convert_to_pinhole_camera_parameters()
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic
    r = cv2.Rodrigues(extrinsic[:3,:3])[0]
    t = extrinsic[:3,3:]
    points_2d, _ = cv2.projectPoints(np.asarray(o3d_pc_r.points), r, t, intrinsic, None)
    correspondence = (points_2d[:,0,:] * 336/1024).astype(int).clip(0, 335)
    visibility = compute_visibility_3d_to_2d(np.asarray(o3d_pc_r.points), correspondence, img_shape=(336, 336))
    cor = np.concatenate((correspondence, visibility[:, np.newaxis]), axis=1)
    np.save(f"{cor_save_path}/view_{view_idx}_cor.npy", cor)
    
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image) *255
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    post_transform = transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC)
    image = post_transform(image)
    image_path = f"{view_save_path}/view_{view_idx}.png"
    image.save(image_path, dpi=(2000,2000))
    vis.destroy_window()
    
    vis_gt = o3d.visualization.Visualizer()
    vis_gt.create_window(width=1036, height=1036, visible=False)
    vis_gt.add_geometry(o3d_pc_gt_r)
    render_option = vis_gt.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])
    render_option.point_size = 6
    render_option.light_on = False
    
    gt_image = vis_gt.capture_screen_float_buffer(do_render=True)
    gt_image = np.asarray(gt_image) * 255
    gt_image = np.clip(gt_image, 0, 255).astype(np.uint8)
    gt_image = Image.fromarray(gt_image)
    gt_image = post_transform(gt_image)
    gt_image_path = f"{gt_save_path}/view_{view_idx}_gt.png"
    gt_image.save(gt_image_path, dpi=(2000, 2000))
    vis_gt.destroy_window()
        
        
def get_mv_images(tiff_path, rgb_path, gt_path, point_size, view_save_path = "./mutilv_o", gt_save_path = "./mutilv_o",cor_save_path = './mutilv_o',file_id='000'):
    transform = transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC)
    organized_pc = read_tiff_organized_pc(tiff_path)
    organized_pc = resize_organized_pc(organized_pc, 336, 336)
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc)
    rgb = Image.open(rgb_path).convert('RGB')
    
    rgb_img = transform(rgb)
    rgb_img = np.array(rgb_img) / 255.0
    unorganized_rgb = rgb_img.reshape(-1, 3)
    
    if os.path.exists(gt_path):
        gt_img = Image.open(gt_path).convert('L')
    else:
        gt_img = Image.new('L', (336, 336), 0)
    gt_img = transform(gt_img)
    gt_img = np.array(gt_img) / 255.0
    gt_img = np.where(gt_img > 0.5, 1., .0)
    unorganized_gt = gt_img.reshape(-1, 1)
    unorganized_gt = np.repeat(unorganized_gt, 3, axis=1)
    
    nonzero_indices = np.nonzero(unorganized_pc[:, 2])[0]
    np.save(f"{cor_save_path}/nonzero_indices.npy", nonzero_indices)
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]  # r = 0.001
    
    selected_gt_colors = unorganized_gt[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(unorganized_pc_no_zeros)
    # o3d_pc.colors = o3d.utility.Vector3dVector(selected_colors)
    o3d_pc.colors = o3d.utility.Vector3dVector(np.asarray(o3d_pc.points) * 0 + 0.7)
    o3d_pc.transform(matrix_x(np.pi))
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))    
    o3d_pc_gt = o3d.geometry.PointCloud()
    o3d_pc_gt.points = o3d.utility.Vector3dVector(unorganized_pc_no_zeros)
    o3d_pc_gt.colors = o3d.utility.Vector3dVector(selected_gt_colors)
    o3d_pc_gt.transform(matrix_x(np.pi))    
    
    pcd_gt_save_path = os.path.join(os.path.dirname(os.path.dirname(view_save_path)),"gt_pcd")
    if not os.path.exists(pcd_gt_save_path):
        os.makedirs(pcd_gt_save_path)
    o3d.io.write_point_cloud(f"{pcd_gt_save_path}/{file_id}.pcd", o3d_pc_gt)
    o3d_pc_gt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    
    view_list_x = [-45.0,-15.0, 0, 15.0, 45.0]       # 9
    view_list_y = [-30 ,-15, 15, 30]   
    # view_list_x = [-144, -108,-72.0, -36.0, 0, 36.0, 72.0, 108.0, 144]     # 9
    view_idx = 0
    for v_x in view_list_x:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        o3d_pc_gt_o = copy.deepcopy(o3d_pc_gt)
        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(v_x), np.radians(0), np.radians(0)])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        o3d_pc_gt_r = o3d_pc_gt_o.rotate(R, center=o3d_pc.get_center())
        render_and_save_point_cloud(o3d_pc_r, o3d_pc_gt_r, view_idx, cor_save_path, view_save_path, gt_save_path, point_size)
        view_idx += 1
    for v_y in view_list_y:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        o3d_pc_gt_o = copy.deepcopy(o3d_pc_gt)
        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(0), np.radians(v_y), np.radians(0)])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        o3d_pc_gt_r = o3d_pc_gt_o.rotate(R, center=o3d_pc.get_center())
        render_and_save_point_cloud(o3d_pc_r, o3d_pc_gt_r, view_idx, cor_save_path, view_save_path, gt_save_path, point_size)
        view_idx += 1

       
def process_directory(directory_path, point_size):
    for root, dirs, files in os.walk(directory_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(".tiff"):
                tiff_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                rgb_path = tiff_path.replace("xyz", "rgb").replace(".tiff", ".png")
                gt_path = tiff_path.replace("xyz", "gt").replace(".tiff", ".png")
                img_save_path = os.path.join(root.replace("xyz","2d_rendering"), file_id)
                gt_save_path = os.path.join(root.replace("xyz", "2d_gt"), file_id)
                cor_save_path = os.path.join(root.replace("xyz", "2d_3d_cor"), file_id)
                
                os.makedirs(img_save_path, exist_ok=True)
                os.makedirs(gt_save_path, exist_ok=True)
                os.makedirs(cor_save_path, exist_ok=True)
                get_mv_images(tiff_path, rgb_path, gt_path, point_size, img_save_path, gt_save_path, cor_save_path, file_id)
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("PointAD", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    args = parser.parse_args()

    cls_list = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 'rope', 'peach', 'foam', 'potato', 'tire']
    point_sizes = [7, 12, 7, 6, 10, 5, 7, 8, 8, 8]
    for cls, point_size in zip(cls_list, point_sizes):
        base_directory = f"{args.data_path}/{cls}"
        for folder in ["test"]:
            directory_path = os.path.join(base_directory, folder)
            process_directory(directory_path, point_size)
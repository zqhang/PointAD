import math
import re
import sys
import os
from .utils import *
import numpy as np
from PIL import Image
from torchvision import transforms
import open3d as o3d
import copy
import cv2

# The same camera has been used for all the images
FOCAL_LENGTH = 711.11

def remove_point_cloud_background(pc):

    # The second dim is z
    dz =  pc[256,1] - pc[-256,1]
    dy =  pc[256,2] - pc[-256,2]

    norm =  math.sqrt(dz**2 + dy**2)
    start_points = np.array([0, pc[-256, 1], pc[-256, 2]])
    cos_theta = dy / norm
    sin_theta = dz / norm

    # Transform and rotation
    rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, -sin_theta],[0, sin_theta, cos_theta]])
    processed_pc = (rotation_matrix @ (pc - start_points).T).T

    # Remove background point
    mask = np.ones(processed_pc.shape[0], dtype=bool)
    for i in range(processed_pc.shape[0]):
        if processed_pc[i,1] > -0.02 or processed_pc[i,2] > 1.8 or processed_pc[i,0] > 1 or processed_pc[i,0] < -1:
            processed_pc[i, :] = -start_points
            mask[i] = False
    kept_indices = np.where(mask)[0]

    processed_pc = (rotation_matrix.T @ processed_pc.T).T + start_points

    index = [0, 2, 1]
    processed_pc = processed_pc[:,index]
    return processed_pc*[0.1, -0.1, 0.1], kept_indices
    
def adjust_view_to_fit_screen(vis, o3d_pc_r, screen_width, screen_height):
    out_screen = True
    while out_screen:
        out_screen = False
        view_control = vis.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        extrinsic = np.array(cam_params.extrinsic)
        view_point = np.array(cam_params.extrinsic[0:3, 3])
        extrinsic[:3, 3] = view_point + np.array([0, 0, 0.005])
        cam_params.extrinsic = extrinsic
        current_params = view_control.convert_to_pinhole_camera_parameters()
        cam_params.intrinsic.width = current_params.intrinsic.width
        cam_params.intrinsic.height = current_params.intrinsic.height
        view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        
        param = view_control.convert_to_pinhole_camera_parameters()
        intrinsic = param.intrinsic.intrinsic_matrix
        extrinsic = param.extrinsic

        r = cv2.Rodrigues(extrinsic[:3, :3])[0]
        t = extrinsic[:3, 3:]
        points_2d, _ = cv2.projectPoints(np.asarray(o3d_pc_r.points), r, t, intrinsic, None)
        points_2d[:, 0, 0] = points_2d[:, 0, 0] * screen_width / 1036
        points_2d[:, 0, 1] = points_2d[:, 0, 1] * screen_height / 1036

        for point in points_2d:
            x, y = point[0]
            if x < 0 or x >= screen_width or y < 0 or y >= screen_height:
                out_screen = True
                break

def render_and_save_point_cloud(o3d_pc_r, o3d_pc_gt_r, view_idx, cor_save_path, view_save_path, gt_save_path, point_size):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1036, height=1036, visible=False)
    vis.add_geometry(o3d_pc_r)
    
    screen_width, screen_height = 1036, 1036
    adjust_view_to_fit_screen(vis, o3d_pc_r, screen_width, screen_height)
    
    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()    
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1, 1, 1])
    render_option.light_on = True
    render_option.point_size = point_size
    render_option.mesh_show_back_face = True
    
    control = vis.get_view_control()
    param = control.convert_to_pinhole_camera_parameters()
    intrinsic = param.intrinsic.intrinsic_matrix
    extrinsic = param.extrinsic
    r = cv2.Rodrigues(extrinsic[:3, :3])[0]
    t = extrinsic[:3, 3:]
    points_2d, _ = cv2.projectPoints(np.asarray(o3d_pc_r.points), r, t, intrinsic, None)
    correspondence = (points_2d[:, 0, :] * 336 / 1036).astype(int).clip(0, 335)
    visibility = compute_visibility_3d_to_2d(np.asarray(o3d_pc_r.points), correspondence, img_shape=(336, 336))
    cor = np.concatenate((correspondence, visibility[:, np.newaxis]), axis=1)
    np.save(f"{cor_save_path}/view_{view_idx}_cor.npy", cor)
    
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    
    post_transform = transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC)
    image = post_transform(image)
    image_path = f"{view_save_path}/view_{view_idx}.png"
    image.save(image_path, dpi=(2000, 2000))
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

    
def get_mv_images(ply_path, gt_path, point_size = 8, view_save_path = "./mutil_views", gt_save_path = "./mutil_views",cor_save_path = './mutil_views', file_id= "00"):
    origin_pc = o3d.io.read_point_cloud(ply_path)
    origin_pc_points = np.asarray(origin_pc.points)
    origin_pc_points = origin_pc_points.reshape(512,512,3)
    origin_pc_points = resize_organized_pc(origin_pc_points, 336, 336)
    origin_pc_points = origin_pc_points.reshape(-1,3)
    # filter the background points
    pc_points, kept_index = remove_point_cloud_background(origin_pc_points)
    np.save(f"{cor_save_path}/nonzero_indices.npy", kept_index)
    pc_points = pc_points.reshape(-1,3)
    pc_points = pc_points[kept_index,:]
    pc_points = pc_points
    origin_pc_colors = np.asarray(origin_pc.colors)
    origin_pc_colors = origin_pc_colors.reshape(512,512,3)
    
    img_ply = (origin_pc_colors * 255).astype(np.uint8)
    img = Image.fromarray(img_ply)
    img_save_path = os.path.join(os.path.dirname(os.path.dirname(view_save_path)),"rgb")
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    img.save(f"{img_save_path}/{file_id}.png")
    origin_pc_colors = resize_organized_pc(origin_pc_colors, 336, 336)
    origin_pc_colors = origin_pc_colors.reshape(-1,3)
    colors = origin_pc_colors[kept_index,:]
    
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(pc_points)
    # o3d_pc.colors = o3d.utility.Vector3dVector(colors)
    o3d_pc.colors = o3d.utility.Vector3dVector(np.asarray(o3d_pc.points) * 0 + 0.7)
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

    gt_img = Image.open(gt_path).convert('L')
    gt_img_save_path = os.path.join(os.path.dirname(os.path.dirname(view_save_path)),"gt")
    if not os.path.exists(gt_img_save_path):
        os.makedirs(gt_img_save_path)
    gt_img.save(f"{gt_img_save_path}/{file_id}.png")
    
    gt_img = np.array(gt_img) / 255.0
    gt_img = np.where(gt_img > 0.5, 1., .0)
    gt_img = resize_organized_pc(gt_img[:, :, np.newaxis], target_height=336, target_width=336, mode = "nearest")
    gt_img = np.repeat(gt_img.reshape(-1,1), 3, axis=1)
    
    gt_img = gt_img[kept_index,:]
    o3d_pc_gt = copy.deepcopy(o3d_pc)
    o3d_pc_gt.colors = o3d.utility.Vector3dVector(gt_img)
    pcd_save_path = os.path.join(os.path.dirname(os.path.dirname(view_save_path)),"pcd")
    if not os.path.exists(pcd_save_path):
        os.makedirs(pcd_save_path)
    o3d.io.write_point_cloud(f"{pcd_save_path}/{file_id}.pcd", o3d_pc)
    
    pcd_gt_save_path = os.path.join(os.path.dirname(os.path.dirname(view_save_path)),"gt_pcd")
    if not os.path.exists(pcd_gt_save_path):
        os.makedirs(pcd_gt_save_path)
    o3d.io.write_point_cloud(f"{pcd_gt_save_path}/{file_id}.pcd", o3d_pc_gt)
    o3d_pc_gt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    o3d_pc.transform(maxtrix_x(0.25*np.pi))
    o3d_pc_gt.transform(maxtrix_x(0.25*np.pi))
    view_list_x = [-30.0,-15.0, 0, 10.0, 20.0]       # 9 views
    view_list_y = [-30 ,-15, 15, 30]
    # view_list_x = [-60.0,-30.0, 0.0, 30.0]
    # view_list_y = [-45, -15, 15.0, 45]
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
            if file.endswith(".ply"):
                ply_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                file_id = re.sub("[^0-9]", "", file_id)
                gt_path = ply_path.replace("xyz", "gt").replace(".ply", ".png").replace("pcd", "mask")
                save_base_path = root.replace("Eyecandies", "Eyecandies_rendered")
                os.makedirs(save_base_path, exist_ok=True)
                if int(file_id) < 1000:
                    save_base_path = os.path.join(save_base_path,"good")
                else:
                    save_base_path = os.path.join(save_base_path,"anomaly")
                render_save_path = os.path.join(os.path.join(save_base_path, "2d_rendering"), file_id)
                gt_save_path = os.path.join(os.path.join(save_base_path, "2d_gt"), file_id)
                cor_save_path = os.path.join(os.path.join(save_base_path, "2d_3d_cor"), file_id)
                
                os.makedirs(render_save_path, exist_ok=True)
                os.makedirs(gt_save_path, exist_ok=True)
                os.makedirs(cor_save_path, exist_ok=True)
                
                get_mv_images(ply_path, gt_path, point_size, render_save_path, gt_save_path, cor_save_path, file_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("PointAD", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    args = parser.parse_args()

    cls_list= ["CandyCane","ChocolateCookie", "ChocolatePraline", "Confetto", "GummyBear", "HazelnutTruffle", "LicoriceSandwich", "Lollipop", "Marshmallow", "PeppermintCandy"]
    for cls in cls_list:
        base_directory = f"{args.data_path}/{cls}"
        point_size = 10
        for folder in ["test"]:
            directory_path = os.path.join(base_directory, folder)
            process_directory(directory_path, point_size)

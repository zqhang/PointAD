import os
from sklearn.neighbors import NearestNeighbors
from .utils import *
import numpy as np
from PIL import Image
from torchvision import transforms
import open3d as o3d
import copy
import cv2

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def render_and_save_point_cloud(o3d_pc_r, o3d_pc_gt_r, view_idx, cor_save_path, view_save_path, gt_save_path, point_size):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1036, height=1036, visible=False)
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
    correspondence = (points_2d[:,0,:] * 336/1036).astype(int).clip(0, 335)
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


def interpolation_points_with_gt_colors(points, gt_colors, target_size):
    num_points = len(points)
    num_new_points = target_size - num_points

    neigh = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(points)
    distances, indices = neigh.kneighbors(points)

    new_points = []
    new_colors = []
    for idx in range(num_points):
        point = points[idx]
        point_color = gt_colors[idx]
        for neighbor_idx in indices[idx, 1:]:
            neighbor_point = points[neighbor_idx]
            neighbor_color = gt_colors[neighbor_idx]

            for ratio in np.linspace(0.25, 0.75, num=3):
                new_point = point * (1 - ratio) + neighbor_point * ratio
                new_points.append(new_point)
                
                if ratio <= 0.5:
                    new_colors.append(point_color)
                else:
                    new_colors.append(neighbor_color)

    if len(new_points) > num_new_points:
        new_points = np.array(new_points)
        new_colors = np.array(new_colors)
        selected_indices = np.random.choice(len(new_points), size=num_new_points, replace=False)
        new_points = new_points[selected_indices]
        new_colors = new_colors[selected_indices]
    return np.vstack((points, new_points)), np.vstack((gt_colors, new_colors))


def get_mv_images(pcd_path, txt_path, point_size, view_save_path = "./mutil_views", gt_save_path = "./mutil_views",cor_save_path = './mutil_views', file_id = "000"):
    target_size = 336*336
    if not os.path.exists(txt_path):
        o3d_pc_o = o3d.io.read_point_cloud(pcd_path)
        points_o = np.asarray(o3d_pc_o.points)
        gt_colors = np.zeros_like(points_o)
    else:
        txt_data = np.genfromtxt(txt_path, delimiter=',')
        points_o = txt_data[:, :3]  # XYZ
        gt_colors = txt_data[:, 3]
        gt_colors = gt_colors.reshape(-1, 1).repeat(repeats=3, axis=-1)
    
    if len(points_o) >= target_size:     
        points_o = points_o.reshape(1, -1, 3)
        points_o = torch.from_numpy(points_o).float()
        if torch.cuda.is_available():
            points_o = points_o.to('cuda')

        points_idx = farthest_point_sample(points_o, target_size)
        new_points = index_points(points_o, points_idx)
        o3d_pc = o3d.geometry.PointCloud()
        points_o = points_o.squeeze(0).cpu().numpy()
        points = new_points.squeeze(0).cpu().numpy()
        o3d_pc.points = o3d.utility.Vector3dVector(points)

        colors = np.asarray(o3d_pc.points) * 0 + 0.7
        o3d_pc.colors = o3d.utility.Vector3dVector(colors)
        
        gt_colors = gt_colors.reshape(1, -1, 3)
        gt_colors = torch.from_numpy(gt_colors).float()
        if torch.cuda.is_available():
            gt_colors = gt_colors.to('cuda')
        gt_colors = index_points(gt_colors, points_idx)
        gt_colors = gt_colors.squeeze(0).squeeze(-1).cpu().numpy()
            
    else:
        points, gt_colors = interpolation_points_with_gt_colors(points_o, gt_colors, target_size)
        o3d_pc = o3d.geometry.PointCloud()
        o3d_pc.points = o3d.utility.Vector3dVector(points)
        o3d_pc.colors = o3d.utility.Vector3dVector(np.asarray(o3d_pc.points) * 0 + 0.7)
    
    o3d_pc_gt = o3d.geometry.PointCloud()
    o3d_pc_gt.points = o3d_pc.points
    o3d_pc_gt.colors = o3d.utility.Vector3dVector(gt_colors)
    
    o3d_pc.transform(matrix_x(np.pi))
    pcd_save_path = os.path.join(os.path.dirname(os.path.dirname(view_save_path)),"pcd")
    if not os.path.exists(pcd_save_path):
        os.makedirs(pcd_save_path, exist_ok=True)
    o3d.io.write_point_cloud(f"{pcd_save_path}/{file_id}.pcd", o3d_pc)
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30))
    
    o3d_pc_gt.transform(matrix_x(np.pi))
    pcd_save_path = os.path.join(os.path.dirname(os.path.dirname(view_save_path)),"gt_pcd")
    if not os.path.exists(pcd_save_path):
        os.makedirs(pcd_save_path, exist_ok=True)
    o3d.io.write_point_cloud(f"{pcd_save_path}/{file_id}.pcd", o3d_pc_gt)
    o3d_pc_gt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30))
    
    view_list_x = [-144, -108,-72.0, -36.0, 0, 36.0, 72.0, 108.0, 144]     # 9 views
    # view_list_x = [-45.0,-15.0, 0, 15.0, 45.0]       # 9 views
    # view_list_y = [-30 ,-15, 15, 30]
    view_idx = 0
    for v_x in view_list_x:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        o3d_pc_gt_o = copy.deepcopy(o3d_pc_gt)
        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(v_x), np.radians(0), np.radians(0)])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        o3d_pc_gt_r = o3d_pc_gt_o.rotate(R, center=o3d_pc.get_center())
        render_and_save_point_cloud(o3d_pc_r, o3d_pc_gt_r, view_idx, cor_save_path, view_save_path, gt_save_path, point_size)
        view_idx += 1
    # for v_y in view_list_y:
    #     o3d_pc_o = copy.deepcopy(o3d_pc)
    #     o3d_pc_gt_o = copy.deepcopy(o3d_pc_gt)
    #     R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(0), np.radians(v_y), np.radians(0)])
    #     o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
    #     o3d_pc_gt_r = o3d_pc_gt_o.rotate(R, center=o3d_pc.get_center())
    #     render_and_save_point_cloud(o3d_pc_r, o3d_pc_gt_r, view_idx, cor_save_path, view_save_path, gt_save_path, point_size)
    #     view_idx += 1

       
def process_directory(directory_path, point_size):
    for root, dirs, files in os.walk(directory_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(".pcd"):
                pcd_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                base_root = root.replace("/Real3D-AD-PCD","/Real3D-rendered")
                os.makedirs(base_root, exist_ok=True)
                
                base_root =os.path.join(base_root, "anomaly")
                os.makedirs(base_root, exist_ok=True)
                txt_path = pcd_path.replace('test', 'GT').replace('.pcd', '.txt')                
                if os.path.exists(txt_path): # anomaly
                    img_save_path = os.path.join(base_root, "2d_rendering")
                    gt_save_path = os.path.join(base_root, "2d_gt")
                    cor_save_path = os.path.join(base_root, "2d_3d_cor")
                    
                    os.makedirs(img_save_path, exist_ok=True)
                    os.makedirs(gt_save_path, exist_ok=True)
                    os.makedirs(cor_save_path, exist_ok=True)
                
                    img_save_path = os.path.join(img_save_path, file_id)
                    gt_save_path = os.path.join(gt_save_path, file_id)
                    cor_save_path = os.path.join(cor_save_path, file_id)
                    
                    os.makedirs(img_save_path, exist_ok=True)
                    os.makedirs(gt_save_path, exist_ok=True)
                    os.makedirs(cor_save_path, exist_ok=True)   
                    get_mv_features(pcd_path, txt_path, point_size, img_save_path, gt_save_path, cor_save_path, file_id)                    
                else:
                    base_root =base_root.replace("anomaly", "good")
                    os.makedirs(base_root, exist_ok=True)
                    
                    img_save_path = os.path.join(base_root, "2d_rendering")
                    gt_save_path = os.path.join(base_root, "2d_gt")
                    cor_save_path = os.path.join(base_root, "2d_3d_cor")
                    
                    os.makedirs(img_save_path, exist_ok=True)
                    os.makedirs(gt_save_path, exist_ok=True)
                    os.makedirs(cor_save_path, exist_ok=True)
                    
                    img_save_path = os.path.join(img_save_path, file_id)
                    gt_save_path = os.path.join(gt_save_path, file_id)
                    cor_save_path = os.path.join(cor_save_path, file_id)
                    
                    os.makedirs(img_save_path, exist_ok=True)
                    os.makedirs(gt_save_path, exist_ok=True)
                    os.makedirs(cor_save_path, exist_ok=True)
                    
                    get_mv_images(pcd_path, txt_path, point_size, img_save_path, gt_save_path, cor_save_path, file_id)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("PointAD", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    args = parser.parse_args()

    cls_list = ['airplane', 'candybar', 'car', 'chicken', 'diamond', 'duck', 'fish', 'gemstone', 'seahorse', 'shell', 'starfish', 'toffees']
    for cls in cls_list:
        base_directory = f"{args.data_path}/{cls}"
        point_size = 12
        for folder in ["test"]:
            directory_path = os.path.join(base_directory, folder)
            process_directory(directory_path, point_size)

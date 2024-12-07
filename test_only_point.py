import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm

import os
import random
import numpy as np
from tabulate import tabulate
from utils import get_transform

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



from visualization import visualizer

from metrics import image_level_metrics, pixel_level_metrics
from tqdm import tqdm
from scipy.ndimage import gaussian_filter


def back_to_3d(d2_similarity_map, d2_3d_cor, non_zero_index, ori_resolution = 336):
    # _, h, w, _ = d2_similarity_map.shape
    h = torch.sqrt(torch.tensor(d2_3d_cor.shape[2])).int()
    w = h
    b, nv, num_points, _ = d2_3d_cor.shape
    xx = d2_3d_cor[:, :, :, 0].reshape(-1).long()
    yy = d2_3d_cor[:, :, :, 1].reshape(-1).long()
    nbatch = torch.repeat_interleave(torch.arange(0, b*nv)[:,None], num_points).reshape(-1, ).cuda().long()
    d2_similarity_map = d2_similarity_map.permute(0, 3, 1, 2)
    point_logits = d2_similarity_map[nbatch, :, yy, xx]
    point_logits = point_logits.reshape(b, nv, num_points, 2)
    vweights = torch.ones((1, nv, 1, 1))
    vweights = vweights.reshape(1, -1, 1, 1).to(point_logits.device)
    is_seen = d2_3d_cor[:, :, :, 2].reshape(b, nv, num_points, 1)
    point_logits = point_logits * vweights * is_seen * non_zero_index
    mask = is_seen.bool() | (~non_zero_index.bool())
    point_logits = point_logits.sum(1)/ (mask.sum(1))
    point_logits = point_logits.reshape(b, h, w, 2)
    return point_logits

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    train_class = args.train_class

    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}
    
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details = AnomalyCLIP_parameters)
    model.eval()

    preprocess, target_transform, target_transform_pc = get_transform(args)
    test_data = Dataset(root=dataset_dir, dataset_name = args.dataset, transform=preprocess, target_transform=target_transform, target_transform_pc = target_transform_pc, mode='test', is_all = True, point_size = args.point_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    # metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        results[obj]['color_pr_sp'] = []
        results[obj]['integrate_pr_sp'] = []

        results[obj]['imgs_masks'] = []
        results[obj]['anomaly_maps'] = []
        results[obj]['color_anomaly_maps'] = []
        results[obj]['integrate_anomaly_maps'] = []

        # metrics[obj] = {}
        # metrics[obj]['pixel-auroc'] = 0
        # metrics[obj]['pixel-aupro'] = 0
        # metrics[obj]['image-auroc'] = 0
        # metrics[obj]['image-ap'] = 0

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer = 20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)


    model.to(device)
    for idx, items in enumerate(tqdm(test_dataloader)):
        # image = items['img'].to(device)
        cls_name = items['cls_name']
        cls_id = items['cls_id']
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]['imgs_masks'].append(gt_mask)  # px
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())


        render_image = items['d2_render_img'].to(device)
        b, nv, c, h, w = render_image.shape
        render_image = render_image.reshape(-1, c, h, w)
        d2_render_anomaly = items['d2_render_anomaly'].to(device)
        d2_3d_cor = items['d2_3d_cor'].to(device)
        non_zero_index = items['non_zero_index'].to(device)
        non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)


############################################################

        with torch.no_grad():
            image_features, patch_features = model.encode_image(render_image, features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]
####################################################
            text_probs = torch.chunk(text_probs, nv, dim = 0)
            text_probs = torch.stack(text_probs, dim = 1).mean(1)
####################################################
            patch_feature = patch_features[0]
            patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
####################################################

            similarity_map_nv = torch.chunk(similarity_map, nv, dim = 0)

            similarity_map_nv = torch.stack(similarity_map_nv, dim = 1)
            anomaly_map_nv = (similarity_map_nv[...,1] + 1 - similarity_map_nv[...,0])/2.0

            # print("text_probs similarity_map_nv anomaly_map_nv", text_probs.shape, similarity_map_nv.shape, anomaly_map_nv.shape)
            #########################################
            d3_similarity_map = back_to_3d(similarity_map, d2_3d_cor, non_zero_index)
            ####################################################
            anomaly_map = d3_similarity_map[...,1]

            ###################################################p#
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma = args.sigma)) for i in anomaly_map.detach().cpu()], dim = 0)

#####################################################################

            results[cls_name[0]]['pr_sp'].extend(0.5*anomaly_map.max() + 0.5*text_probs.detach().cpu())

          
            results[cls_name[0]]['anomaly_maps'].append(anomaly_map)


            # visualizer(items['img_path'], integrate_anomaly_maps.detach().cpu().numpy(), args.image_size, args.save_path, cls_name)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []

    obj_list = [c for c in obj_list if c != train_class]
    for obj in obj_list:
        table = []
        color_table = []
        integrate_table = []
        table.append(obj)
        color_table.append(obj)
        integrate_table.append(obj)
        results[obj]['imgs_masks'] = torch.cat(results[obj]['imgs_masks'])
        results[obj]['anomaly_maps'] = torch.cat(results[obj]['anomaly_maps']).detach().cpu().numpy()
        if args.metrics == 'image-level':
            #### point
            image_auroc = image_level_metrics(results, obj, "image-auroc", modality = 'pr_sp')
            image_ap = image_level_metrics(results, obj, "image-ap", modality = 'pr_sp')
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == 'pixel-level':
            #### point
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", modality = 'anomaly_maps')
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            pixel_auroc_list.append(pixel_auroc)
        elif args.metrics == 'image-pixel-level':
            #### point
            image_auroc = image_level_metrics(results, obj, "image-auroc", modality = 'pr_sp')
            image_ap = image_level_metrics(results, obj, "image-ap", modality = 'pr_sp')
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc", modality = 'anomaly_maps')
            table.append(str(np.round(pixel_auroc * 100, decimals=1)))
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap) 
            pixel_auroc_list.append(pixel_auroc)
        table_ls.append(table)
    if args.metrics == 'image-level':
        # logger
        table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")


    elif args.metrics == 'pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))
                       ])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'pixel_aupro'], tablefmt="pipe")


    elif args.metrics == 'image-pixel-level':
        # logger
        table_ls.append(['mean', str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
        results = tabulate(table_ls, headers=['objects', 'pixel_auroc', 'image_auroc', 'image_ap'], tablefmt="pipe")

    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PointAD", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/', help='path to checkpoint')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    parser.add_argument("--feature_map_layer", type=int,  nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--metrics", type=str, default='image-pixel-level')
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="zero shot")
    parser.add_argument("--train_class", type=str, default="cookie", help="zero shot or few shot")
    parser.add_argument("--point_size", type=int, default=336, help="save frequency")
    
    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)

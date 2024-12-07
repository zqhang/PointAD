import os
import json

def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]
    # return [
      
    #     "peach",

    # ]
class MVTecSolver(object):
    # CLSNAMES = [
    #     'bottle', 'cable', 'capsule', 'carpet', 'grid',
    #     'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    #     'tile', 'toothbrush', 'transistor', 'wood', 'zipper',
    # ]
    CLSNAMES = mvtec3d_classes()
    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/all_meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            # for phase in ['train', 'test']:
            for phase in ['test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    # img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    img_path = f'{cls_dir}/{phase}/{specie}'
                    print('img_path', img_path)
                    d2_img_names = os.listdir(os.path.join(img_path, 'rgb'))
                    d2_mask_names = os.listdir(os.path.join(img_path, 'gt')) if is_abnormal or phase == 'test' else None
                    d3_pc = os.listdir(os.path.join(img_path, 'xyz'))
                    d2_img_names.sort()
                    d2_mask_names.sort() if d2_mask_names is not None else None
                    d3_pc.sort()
                    for idx, d2_img_name in enumerate(d2_img_names):
                        info_img = dict(
                            d2_img_path=f'{img_path}/rgb/{d2_img_name}',
                            d2_mask_path=f'{img_path}/gt/{d2_mask_names[idx]}' if d2_mask_names else '',
                            d3_pc=f'{img_path}/xyz/{d3_pc[idx]}',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                            d2_render_img_path=f'{img_path}/2d_rendering/{d2_img_name.split(".")[0]}',
                            d2_render_gt_path=f'{img_path}/2d_gt/{d2_img_name.split(".")[0]}',
                            d2_corrdinate=f'{img_path}/2d_3d_cor/{d2_img_name.split(".")[0]}',
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)
if __name__ == '__main__':
    runner = MVTecSolver(root='/remote-home/iot_zhouqihang/data/mvtec_3d_mv/mvtec_3d_9_views')
    runner.run()

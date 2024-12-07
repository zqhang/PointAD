#!/bin/bash

cls_ids=(0 1 2 3 4 5 6 7 8 9 10 11)
for cls_id in "${!cls_ids[@]}";do
    python mvtec_3d_anomaly_real_3d_ad_class_specific.py --cls_id ${cls_ids[cls_id]}
    wait
done

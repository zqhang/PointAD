
device=1

obj_list=("cookie" "carrot" "dowel")
# obj_list=("carrot" "dowel")
cls_ids=(0 1 2)
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
        ## train on the VisA dataset
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_mvtec_3d
            save_dir=./exps_${base_dir}_336_4/${obj_list[cls_id]}/
            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset mvtec_pc_3d_rgb --train_data_path /remote-home/iot_zhouqihang/data/mvtec_3d_mv/mvtec_3d_9_views \
            --save_path ${save_dir} \
            --features_list 24 --image_size 336  --batch_size 4 --print_freq 1 \
            --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_dataset_name ${obj_list[cls_id]}
            
            CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec_pc_3d_rgb \
            --data_path /remote-home/iot_zhouqihang/data/mvtec_3d_mv/mvtec_3d_9_views --save_path ./results/mvtec_${base_dir}_pc_336_4_all/with_color_max_scor${obj_list[cls_id]}/zero_shot_1 \
            --checkpoint_path ${save_dir}epoch_15.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_class ${obj_list[cls_id]}
        wait
        done
    done
done

obj_list=('PeppermintCandy' 'LicoriceSandwich' 'Confetto')
# obj_list=("carrot" "dowel")
cls_ids=(0 1 2)
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
        ## train on the VisA dataset
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_eye_3d
            save_dir=./exps_${base_dir}_336_4/${obj_list[cls_id]}/
            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset eye_pc_3d_rgb  --train_data_path /remote-home/iot_zhouqihang/data/Eyecandies_processed \
            --save_path ${save_dir} \
            --features_list 24 --image_size 336  --batch_size 4 --print_freq 1 \
            --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_dataset_name ${obj_list[cls_id]}
            

            CUDA_VISIBLE_DEVICES=${device} python test.py --dataset eye_pc_3d_rgb  \
            --data_path /remote-home/iot_zhouqihang/data/Eyecandies_processed --save_path ./results/mvtec_${base_dir}_pc_336_4_all/with_color_max_scor${obj_list[cls_id]}/zero_shot \
            --checkpoint_path ${save_dir}epoch_15.pth \
            --features_list 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_class ${obj_list[cls_id]}
        wait
        done
    done
done


obj_list=('shell' 'starfish' 'seahorse')
obj_list=('seahorse')
cls_ids=(0 1 2) 
for cls_id in "${!cls_ids[@]}";do
    LOG=${save_dir}"res.log"
    echo ${LOG}
    echo ${cls_id} 
    depth=(9)
    n_ctx=(12)
    t_n_ctx=(4)
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
        ## train on the VisA dataset
            base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_mv9_real_3d
            save_dir=./exps_${base_dir}_336_4/${obj_list[cls_id]}/
            CUDA_VISIBLE_DEVICES=${device} python train.py --dataset real_pc_3d_rgb  --train_data_path /remote-home/iot_zhouqihang/data/Real3D-AD \
            --save_path ${save_dir} \
            --features_list 6 12 18 24 --image_size 336  --batch_size 4 --print_freq 1 \
            --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_dataset_name ${obj_list[cls_id]}
            

            CUDA_VISIBLE_DEVICES=${device} python test_only_point.py --dataset real_pc_3d_rgb  \
            --data_path /remote-home/iot_zhouqihang/data/Real3D-AD --save_path ./results/mvtec_${base_dir}_pc_336_4_all/with_color_max_scor${obj_list[cls_id]}/zero_shot \
            --checkpoint_path ${save_dir}epoch_15.pth \
            --features_list 6 12 18 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --train_class ${obj_list[cls_id]}
        wait
        done
    done
done



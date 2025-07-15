python3 -m modules.training.train \
    --training_type xfeat_default  \
    --megadepth_root_path ../phoenix/S6/zl548/ \
    --synthetic_root_path ../coco_20k \
    --ckpt_save_path weights/2 \
    --batch_size 16 
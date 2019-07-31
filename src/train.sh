#!/bin/bash

set -e
set -x

python main.py ctdet --exp_id pascal_vgg_384 \
                     --arch vgg \
                     --dataset pascal \
                     --num_epochs 70 \
                     --lr_step 45,60 \
                     --input_res 384 \
                     --down_ratio 4 \
                     --gpus 0

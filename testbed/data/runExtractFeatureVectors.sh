#!/bin/bash

CAMERA=camera5
FEAT_SAVE_PATH=features/${CAMERA}

if [ ! -d "$FEAT_SAVE_PATH" ]; then
    mkdir -p $FEAT_SAVE_PATH
fi

./ExtractFeatureVectors.py --model_weight_file model/mobilenetV2.pt --bbox_mat DukeMTMC/detections/bounding_boxes/${CAMERA}_cropped_body.mat --image_dir DukeMTMC/frames/${CAMERA} --saved_feature_mat_path $FEAT_SAVE_PATH --batch_size 32 --num_workers 32 --sys_device_ids [1]

#!/bin/bash

python3 ExtractFeatureVectors.py --model_weight_file model/mobilenetV2.pt --bbox_mat DukeMTMC/detections/bounding_boxes/camera5_cropped_body.mat --image_dir DukeMTMC/frames/camera5 --saved_feature_mat_path features/camera5 --batch_size 512 --num_workers 32 --sys_device_ids [1]

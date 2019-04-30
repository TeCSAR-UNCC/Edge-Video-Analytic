#!/bin/bash

python ExtractFeatureVectors.py --model_weight_file model/mobilenetV2.pt --bbox_mat DukeMTMC/detections/bounding_boxes/camera5_full_body.mat --image_dir DukeMTMC/frames/camera5 --saved_feature_mat_path features/camera5 --batch_size 512 --sys_device_ids [1]

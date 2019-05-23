#!/bin/bash

CAMERA="camera2"

KEYPOINT_SAVE_PATH=$EDGE_ANALYTICS/testbed/data/keypoints/${CAMERA}

if [ ! -d "$KEYPOINT_SAVE_PATH" ]; then
    mkdir -p $KEYPOINT_SAVE_PATH
fi

cd $EDGE_ANALYTICS
#$EDGE_ANALYTICS/build/examples/openpose/openpose.bin -image_dir $EDGE_ANALYTICS/testbed/data/DukeMTMC/frames/$CAMERA/ -render_pose 0 -keypoint_scale 3 -num_gpu 1 -num_gpu_start 1 -write_json $KEYPOINT_SAVE_PATH -display 0

cd $EDGE_ANALYTICS/testbed/data
#matlab -nodisplay -nosplash -nodesktop -r "ExtractKeypointsMat('${CAMERA}'); exit"
matlab -nodisplay -nosplash -nodesktop -r "ExtractBoundingBoxes('${CAMERA}'); exit"

FEAT_SAVE_PATH=DukeMTMC/detections/features/${CAMERA}

if [ ! -d "$FEAT_SAVE_PATH" ]; then
    mkdir -p $FEAT_SAVE_PATH
fi

./ExtractFeatureVectors.py --model_weight_file model/mobilenetV2.pt --bbox_mat DukeMTMC/detections/bounding_boxes/${CAMERA}_full_body.mat --image_dir DukeMTMC/frames/${CAMERA} --saved_feature_mat_path $FEAT_SAVE_PATH --batch_size 32 --num_workers 32 --sys_device_ids [1]

matlab -nodisplay -nosplash -nodesktop -r "CombineFeatureMats('${CAMERA}'); exit"

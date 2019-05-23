#!/bin/bash

CAMERA="camera2"

KEYPOINT_SAVE_PATH=$EDGE_ANALYTICS/testbed/data/keypoints/${CAMERA}

if [ ! -d "$KEYPOINT_SAVE_PATH" ]; then
    mkdir -p $KEYPOINT_SAVE_PATH
fi

$EDGE_ANALYTICS/build/examples/openpose/openpose.bin -image_dir $EDGE_ANALYTICS/testbed/data/DukeMTMC/frames/$CAMERA/ -render_pose 0 -keypoint_scale 3 -num_gpu 1 -write_json $KEYPOINT_SAVE_PATH -display 0

#!/bin/bash

CAMERA="camera5"

for video in $EDGE_ANALYTICS/testbed/data/DukeMTMC/videos/$CAMERA/*.MTS; do
	echo "$video"
	$EDGE_ANALYTICS/build/examples/openpose/openpose.bin -video "$video" -render_pose 0 -keypoint_scale 3 -num_gpu 1 -write_json $EDGE_ANALYTICS/testbed/data/keypoints/ -display 0 -num_gpu_start 1
done;

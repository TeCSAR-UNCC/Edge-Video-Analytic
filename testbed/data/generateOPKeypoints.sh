#!/bin/bash

$EDGE_ANALYTICS/build/examples/openpose/openpose.bin -video $EDGE_ANALYTICS/testbed/data/DukeMTMC/videos/camera5/00000.MTS -render_pose 0 -keypoint_scale 3 -num_gpu 1 -write_json $EDGE_ANALYTICS/testbed/data/keypoints/ -display 0

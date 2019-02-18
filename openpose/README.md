This is the readme to help yall with using the custom code for openpose.

# Pre req
1) Download openpose here: https://github.com/CMU-Perceptual-Computing-Lab/openpose

2) Put the custom code in examples/user_code

# Make
3.1) If you're using the Xavier, place the Makefile and Makefile.config in the base directory for openpose. (Good luck with the dependencies! NOTE: Same dependencies for caffe)

3.2) If you're using the TX2 (Assuming Jetpack 3.3), use: bash ./scripts/ubuntu/install_caffe_and_openpose_JetsonTX2_JetPack3.3.sh

4) make

5) Use this line of code to run it: ./build/examples/user_code/op_bounding_boxes_SD.bin -camera_resolution 640x480 -net_resolution 128x96

If it breaks even though it compiles, I don't know it might be a dependencies thing or ya broke something.

If you want the information about the different types of flags that are available, go to: /include/openpose/flags.hpp
Or run: ./build/examples/openpose/openpose.bin --help


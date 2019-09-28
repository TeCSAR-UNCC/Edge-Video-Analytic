# NVDA_DETEC
# Infrastructure for Real-Time Edge Object Reidentification and Tracking

Detection and tracking infrastructure built on top of OpenPose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)


Note: Path names are setup for “tecsar” by default

## Part 1: Installing Nvidia SDK Manager
1. Go to https://developer.nvidia.com

1. Create an account and log in.

1. Following logging in, go to https://developer.nvidia.com/embedded/jetpack-archive, then click on Jetpack 4.2.2.

1. Scroll down until you see a green button labeled “Download SDK Manager”, then click it.

1. Open a terminal where the .deb file is downloaded to. Run the command:
 sudo apt install sdkmanager_0.9.14-4961_amd64.deb

1. After it completes the installation, type sdkmanager in the terminal. This will open the graphical interface for the SDK Manager. Log in with your NVIDIA account.


## Part 2: Installing prerequisites
1. In the GUI, make sure the target hardware is the Jetson AGX Xavier, and the target operating system is JetPack 4.2.2.

1. Host Machine can be left checked or unchecked (check indicates that files will be downloaded onto your machine before being installed on the Xavier, potentially saving time in the event you need to reload it later). Click Continue.

1. Check all available options, then accept the terms and conditions, then click continue.

1. Connect the Xavier to your PC with a USB-C to USB connector. Follow the instructions on screen. If you select “automatic” during the flashing process, you will need to enter the current username and password for the Xavier.

1. Allow some time for the OS to flash. After the OS is done installing, you will need to manually finish the installation process for it (selecting a username, password, timezone, etc). As soon as it gets to the desktop or login screen, the process will resume.

1. Watch the computer connected to the Xavier. The installation process is not complete until the NVIDIA SDK Manager indicates it.
	1. If issues occur with OpenPose, reinstall CUDA on the Xavier from the Jetpack SDK Manager. Make sure the OS option is unchecked for this process.


## Part 3: Installing OpenPose and REVAMP2T software
1. Download the UNCC OpenPose GitHub repository by running the following command in a terminal:
	git clone https://github.com/TeCSAR-UNCC/Edge-Video-Analytic --recurse-submodules
1. Fix a variable in the trt_helper.hpp file:
Go to Edge-Video-Analytic/examples/user_code
Open the file trt_helper.hpp in a text editor:
On line 10 of the file change “static Logger gLogger“ to “Logger gLogger“ and save.

1. Run the Xavier Setup Script:
In a terminal, change directory to or open a terminal in the root directory (.../Edge-Video-Analytic), then run the following commands:
chmod +x Xavier_Setup.sh
	./Xavier_Setup.sh

1. Test if OpenPose is working by running the following command in a terminal:
		./build/examples/openpose/openpose.bin --video examples/media/video.avi
		More demo examples can be found here

1. Fix a path in the trt_help.hpp file:
Go to Edge-Video-Analytic/examples/user_code
Open the file trt_helper.hpp in a text editor:
		On lines 18 & 19, change the last file path from “/home/tecsar” to “~/”

1. Run make on the files:
In a terminal, change directories to or open a terminal in the root directory (.../Edge-Video-Analytic)
Run the following command in the terminal:
make -j


## Part 4: Running the Server
1. Run the EdgeServer on another Computer (Not the Xavier):
Repeat part 3 on another computer to download the UNCC GitHub repository.

1. Install OpenCV2 by doing the following:
In a terminal, run the following command:
git clone https://github.com/jayrambhia/Install-OpenCV
In a terminal, change directories to or open a terminal to the directory (.../Install-OpenCV/Ubuntu), then run the following commands in a terminal:
	chmod +x *
		./opencv_latest.sh


1. Change directories to or open a terminal to the directory (.../Edge-Video-Analytic/SocketCode), then run the following command:
ifconfig
make all
./EdgeServer.bin

**Remember inet IP address (shown as a red block in the image below), you will need it for step 23:**

![inet_IP](/doc/media/tecsar/inet_IP.png)
1. On the Xavier, change directories to in a terminal or open a terminal in the root directory (.../Edge-Video-Analytic), then open the run file in a text editor.

1. Change the contents after -server_ip (shown as a red block below) to the inet IP address you got in step 21.

![run_change](/doc/media/tecsar/run-script.png)

If you have multiple Xaviers, each with a camera, make sure that each camera has a different camera ID before doing step 24. (see below)

To change the camera ID on each Xavier (skip this if you only have 1 camera):
Go to Edge-Video-Analytic/examples/user_code
Open the file reid_constants.hpp in a text editor:
On line 5, change the value of CAMERA_ID to a number unique to that Xavier.




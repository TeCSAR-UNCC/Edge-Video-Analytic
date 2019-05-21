#!/usr/bin/python3

import cv2
import numpy as np
import glob, os, re
import argparse

class Config(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--cam_num', type=eval, default=(5))

        args = parser.parse_args()
        self.cam_num = args.cam_num

def main():
    cfg = Config()

    start_frames = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];

    cameraName = "camera" + str(cfg.cam_num)

    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    clear = lambda: os.system('clear')

    currentFrame = start_frames[cfg.cam_num - 1]

    for file in sorted(glob.glob(os.path.join("DukeMTMC/videos/" + cameraName + "/*.MTS")), key=numericalSort):
        print (file)
        cap = cv2.VideoCapture(file)

        try:
            if not os.path.exists("DukeMTMC/frames/" + cameraName):
                os.makedirs("DukeMTMC/frames/" + cameraName)
        except OSError:
            print ('Error: Creating directory of data')

        # Capture frame-by-frame
        ret, frame = cap.read()
        while(ret):
            # Saves image of the current frame in jpg file
            name = "DukeMTMC/frames/" + cameraName + "/" + "{:06d}".format(currentFrame) + '.jpg'
            clear()
            print (file)
            print ('Creating...' + name)
            cv2.imwrite(name, frame)

            # To stop duplicate images
            currentFrame += 1

            # Capture frame-by-frame
            ret, frame = cap.read()

        # When everything done, release the capture
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

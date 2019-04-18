import cv2
import numpy as np
import glob, os, re

cameraName = "camera5"

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

clear = lambda: os.system('clear')

currentFrame = 1

for file in sorted(glob.glob(os.path.join("DukeMTMC/videos/" + cameraName + "/*.MTS")), key=numericalSort):
    print file
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
        print file
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

        # Capture frame-by-frame
        ret, frame = cap.read()

    # When everything done, release the capture
    cap.release()
cv2.destroyAllWindows()

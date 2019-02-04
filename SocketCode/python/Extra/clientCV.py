# ## From https://stackoverflow.com/questions/30988033/sending-live-video-frame-over-network-in-python-opencv
# import socket
from numpysocket import NumpySocket

npSocket = NumpySocket()
npSocket.startClient(9999)

# Read until video is completed
while(True):
    # Capture frame-by-frame
    frame = npSocket.recieveNumpy()
    print(frame)


npSocket.endServers()
print("Closing")

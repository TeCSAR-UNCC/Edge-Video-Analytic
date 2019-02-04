#!/usr/bin/env python3

import socket
import numpy as np
from io import BytesIO

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

def recieveNumpy(socket):
    length = None
    ultimate_buffer=b''
    while True:
        data = socket.recv(1024)
        ultimate_buffer += data
        if len(ultimate_buffer) == length:
            break
        while True:
            if length is None:
                if ':' not in ultimate_buffer:
                    break
                # remove the length bytes from the front of ultimate_buffer
                # leave any remaining bytes in the ultimate_buffer!
                length_str, ignored, ultimate_buffer = ultimate_buffer.partition(':')
                length = int(length_str)
            if len(ultimate_buffer) < length:
                break
            # split off the full message from the remaining bytes
            # leave any remaining bytes in the ultimate_buffer!
            ultimate_buffer = ultimate_buffer[length:]
            length = None
            break
    final_image = np.load(BytesIO(ultimate_buffer))['frame']
    print('frame received')
    return final_image


serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serversocket.bind(('localhost', PORT))
serversocket.listen(5)

(clientsocket, address) = serversocket.accept()
print('Connected by', address)

array = recieveNumpy(serversocket)
print(array)

#while 1:
    #accept connections from outside
#    (clientsocket, address) = serversocket.accept()
#    print('Connected by', address)
    #now do something with the clientsocket
    #in this case, we'll pretend this is a threaded server
#    total_data=[]
#    while(len(total_data) < 128):  
      #data = clientsocket.recv(128)
#      total_data.append(data)
#      print(len(total_data))
#      print(total_data)
#      if not data:
#          break
#      clientsocket.sendall(data)

#with conn:
#    print('Connected by', addr)
#    while True:
#        data = conn.recv(1024)
#        if not data:
#            break
#       conn.sendall(data)
#https://docs.python.org/2/howto/sockets.

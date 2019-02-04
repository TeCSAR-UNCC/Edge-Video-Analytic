#!/usr/bin/env python3

import socket
import numpy as np
from io import BytesIO

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server
ARRAY = np.ones(5)

def sendNumpy(socket, image):
    if not isinstance(image, np.ndarray):
        print('not a valid numpy image')
        return
    f = BytesIO()
    np.savez_compressed(f,frame=image)
    f.seek(0)
    out = f.read()
    try:
        socket.sendall(out)
    except Exception:
        exit()
    print('image sent')

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientsocket.connect((HOST, PORT))
#clientsocket.sendall(b'Hello, world')
#clientsocket.sendall(b'2Hello, world')
#clientsocket.sendNumpy(ARRAY)

sendNumpy(clientsocket, ARRAY)

#clientsocket.shutdown(1)
data = clientsocket.recv(1024)
clientsocket.close()
print('Received', repr(data))




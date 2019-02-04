import socket
import numpy as np
from uniqueID import *
from io import BytesIO
import pickle

server_address = '127.0.0.1'  # The server's hostname or IP address
image = np.ones(5)
PORT = 65432

objectA = uniqueID()

if not isinstance(image,np.ndarray):
    print('not a valid numpy image')
client_socket=socket.socket()
try:
    client_socket.connect((server_address, PORT))
    print('Connected to %s on port %s' % (server_address, PORT))
except(socket.error,e):
    print('Connection to %s on port %s failed: %s' % (server_address, PORT, e))
#f = BytesIO()
#np.savez_compressed(f,frame=image)
#f.seek(0)
#out = f.read()

objectA = uniqueID()
#objectA.features = out
objectA.features = image
objectA.name = "A"

# Pickle the object and send it to the server
data_string = pickle.dumps(objectA)

client_socket.sendall(data_string)
print('image sent')
client_socket.shutdown(1)

data = client_socket.recv(1024)
print('Received', repr(data))

client_socket.close()

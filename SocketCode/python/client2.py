import socket
import numpy as np
from uniqueID import *
from io import BytesIO
import pickle

def sendObject(image, name, cameraID):
	if not isinstance(image,np.ndarray):
	    print('not a valid numpy image')
	client_socket=socket.socket()
	try:
	    client_socket.connect((server_address, PORT))
	    print('Connected to %s on port %s' % (server_address, PORT))
	except(socket.error,e):
	    print('Connection to %s on port %s failed: %s' % (server_address, PORT, e))

	objectA = uniqueID()
	objectA.features = image
	objectA.name = name
	objectA.cameraID = cameraID

	# Pickle the object and send it to the server
	data_string = pickle.dumps(objectA)

	client_socket.sendall(data_string)
	print('image sent')
	client_socket.shutdown(1)

	data = client_socket.recv(1024)
	print('Received', repr(data))

	client_socket.close()

server_address = '127.0.0.1'  # The server's hostname or IP address
image = np.ones(5)
PORT = 65432
cameraID = "A"
image = np.random.rand(10,255)

for i in range(0, len(image)):
	sendObject(image[i], str(i), cameraID)

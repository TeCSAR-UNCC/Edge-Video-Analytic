import socket
import numpy as np
from uniqueID import *
from io import BytesIO
import pickle

HOST = 'localhost'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

server_socket=socket.socket() 
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print('waiting for a connection...')

while True:
    client_connection,client_address=server_socket.accept()
    print('connected to ',client_address)

    ultimate_buffer=b''
    receiving_buffer = client_connection.recv(1024)
    if not receiving_buffer: break
    ultimate_buffer+= receiving_buffer
    print('-'),
    data_variable = pickle.loads(ultimate_buffer)
    print(data_variable.name)
    #final_image=np.load(BytesIO(data_variable.features))['frame']
    final_image = data_variable.features
    print(final_image)
    client_connection.sendall(b'Recieved')
    client_connection.close()
server_socket.shutdown(1)
server_socket.close()
print('\nframe received')

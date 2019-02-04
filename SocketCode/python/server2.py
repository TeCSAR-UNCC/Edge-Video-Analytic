import socket
import numpy as np
from uniqueID import *
from io import BytesIO
import pickle
from skimage.measure import compare_ssim as ssim

HOST = 'localhost'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)



###################
img1 = np.zeros(255);
img2 = np.ones(255);
img3 = np.random.rand(255);
img4 = np.random.rand(255);

#table = np.zeros(10,255);

initID = uniqueID();
initID.features = np.zeros(255)


listID = []
for i in range (0,10):
    listID.append(initID)

result = np.zeros(10);

#result = ssim(img1,img1);
#print(result)

###########################

server_socket=socket.socket() 
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print('waiting for a connection...')

while True:
    client_connection,client_address=server_socket.accept()
    print('connected to ',client_address)
    ultimate_buffer=b''

    while True:
        receiving_buffer = client_connection.recv(1024)
        if not receiving_buffer: break
        ultimate_buffer+= receiving_buffer

    print('-'),
    data_variable = pickle.loads(ultimate_buffer)
    #print(data_variable.name)
    final_image = data_variable.features
    #print(final_image)

    for x in range(0,10):
      result[x] = ssim(listID[x].features,final_image);

    max_idx = np.argmax(result)
    if(result[max_idx]>SSIM_THRESH):
        listID[max_idx] = data_variable
      
    print(result)
    client_connection.sendall(b'Recieved')
    client_connection.close()
server_socket.shutdown(1)
server_socket.close()
print('\nframe received')

from numpysocket import NumpySocket

host_ip = 'localhost'  # change me
npSocket = NumpySocket()
npSocket.startServer(host_ip, 9999)

array = np.zeros(5);
npSocket.sendNumpy(array)

# When everything done, release the video capture object
npSocket.endServer()

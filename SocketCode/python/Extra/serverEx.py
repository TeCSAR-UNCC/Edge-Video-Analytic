import socket

s = socket.socket()

port = 20000

s.bind(('',port))

s.listen(5)

c, addr = s.accept()
c.send("Sent from Server")
print c.recv(1024)
c.close



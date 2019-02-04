import socket
import numpy as np
import thread as thread
import ctypes
import time

database = []
names = []


def handleConn(commIndex):
  #Thread
  slot = server.cRecv(commIndex)

  name = server.getID(slot)
  camera = server.getCamera(slot)
  xpos = server.getXPos(slot)
  ypos = server.getYPos(slot)
  ht = server.getHeight(slot)
  wd = server.getWidth(slot)
  clear = server.clearSlot(slot)

  send = 0
  sendName = 0
  #compare
  if(xpos in database):
    index = database.index(xpos)
    sendName = int(names[index])
    send = 1
    print("Old")
  else:
    database.append(xpos)
    names.append(name)
    print("New")
  #append, notify, or update
  #if notify, send
  #time.sleep(5)
  if(send==1):  
    send = server.cSend(commIndex, ctypes.c_int(sendName))
  #else, close
  cClose = server.cCloseComm(commIndex)
  print("Closed Thread");
  #end thread



#load the shared object file
#server = CDLL('./server.so')
server = ctypes.CDLL('libServer.so')
#/home/matias/Desktop/socketProg/python/

#Find sum of integers
init = server.serverInit()

#listen = server.cListen()
#accept = server.cAccept()

while(1):
  commIndex = server.cListenAccept()
  if(commIndex >= 0):
    thread.start_new_thread(handleConn, (commIndex, ))



lClose = server.cCloseListen()


print(name, camera, xpos, ypos, ht, wd)









import socket
import numpy as np
#import thread
import ctypes




#load the shared object file
#server = CDLL('./server.so')
server = ctypes.CDLL('libServer.so')
#/home/matias/Desktop/socketProg/python/

#Find sum of integers
init = server.serverInit()

#listen = server.cListen()
#accept = server.cAccept()

commIndex = server.cListenAccept()

if(commIndex >= 0):
  #Start Thread
  slot = server.cRecv(commIndex)

  name = server.getID(slot)
  camera = server.getCamera(slot)
  xpos = server.getXPos(slot)
  ypos = server.getYPos(slot)
  ht = server.getHeight(slot)
  wd = server.getWidth(slot)
  clear = server.clearSlot(slot)

  #compare
  #append, notify, or update
  #if notify, send
  send = server.cSend(commIndex)
  #else, close
  cClose = server.cCloseComm(commIndex)
  #end thread



lClose = server.cCloseListen()


print(name, camera, xpos, ypos, ht, wd)









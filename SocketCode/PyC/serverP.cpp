#include <unistd.h> 
#include <stdio.h> 
#include <sys/socket.h> 
#include <stdlib.h> 
#include <netinet/in.h> 
#include <string.h> 
#define PORT 65432
#define IPADDR INADDR_ANY

#include "opencv2/opencv.hpp"
using namespace cv;
 //g++ serverTest.cpp -o server `pkg-config --cflags --libs opencv`

//gcc -shared -Wl,-soname,adder -o adder.so -fPIC add.c `pkg-config --cflags --libs opencv`
//g++ -shared -o libServer.so -fPIC serverP.cpp `pkg-config --cflags --libs opencv`

struct Info{
  int name;
  int cameraID;
  int cx;
  int cy;
  int h;
  int w;
};

struct uniqueID{
  Info uInfo;
  //Mat hist;
  //uint8_t hits;
  //time
};

#define TABLESIZE 300
#define NUMSTRUCTELEMENTS 6
#define SLOTFULL 1
#define SLOTEMPTY 0
#define NUMCOMMSOCK 100
//struct declaration and allocation
int recvTable[TABLESIZE][NUMSTRUCTELEMENTS+1];
int comm_fd[NUMCOMMSOCK][2];

int listen_fd, comm;


extern "C" int serverInit() {

  for(int i = 0; i < TABLESIZE; ++i) {
    recvTable[i][NUMSTRUCTELEMENTS] = SLOTEMPTY;
  }

	struct sockaddr_in servaddr;

	listen_fd = socket(AF_INET, SOCK_STREAM, 0);

	bzero( &servaddr, sizeof(servaddr));

	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = htons(IPADDR);
	servaddr.sin_port = htons(PORT);

	bind(listen_fd, (struct sockaddr *) &servaddr, sizeof(servaddr));


return 0;
}


/*
thread
  listen
  accept
  return the comm_fd to shift register

thread
the recv on that comm_fd (pop from shift register)
send()
close comm_fd

*/


extern "C" int cListen() {
	listen(listen_fd, 10);
  return 0;
}

extern "C" int cAccept() {
  comm = accept(listen_fd, (struct sockaddr*) NULL, NULL);
  return 0;
}

extern "C" int cListenAccept() {
	listen(listen_fd, 10);
  comm = accept(listen_fd, (struct sockaddr*) NULL, NULL);

  int commIndex = -1;
  for(int i = 0; i<NUMCOMMSOCK; ++i) {
    if(comm_fd[i][1] == SLOTEMPTY) {
      comm_fd[i][0] = comm;
      commIndex = i;
      comm_fd[i][1] = SLOTFULL;
      break;
    }
  }

  if(commIndex < 0) {
    printf("Error: comm_fd buffer is full\n");
    close(comm);
  }

  return commIndex;
}


//returns -1 if no slot found
extern "C" int cRecv(int commIndex) {
  //struct declaration and allocation
	struct Info recvStruct;
  int status = 1;
  //while(1) {
    status = recv(comm_fd[commIndex][0], &recvStruct, sizeof(recvStruct), 0);
    printf("Recieved: name: %d cx: %d\n", recvStruct.name, recvStruct.cx);

    int slot = -1;
    for(int i = 0; i < TABLESIZE; ++i) {
      if(recvTable[i][NUMSTRUCTELEMENTS] == SLOTEMPTY) {
        slot = i;
        break;
      }
    }
    if(slot >= 0) {
      //fill slot
      recvTable[slot][0] = recvStruct.name;
      recvTable[slot][1] = recvStruct.cameraID;
      recvTable[slot][2] = recvStruct.cx;
      recvTable[slot][3] = recvStruct.cy;
      recvTable[slot][4] = recvStruct.h;
      recvTable[slot][5] = recvStruct.w;
      recvTable[slot][NUMSTRUCTELEMENTS] = SLOTFULL;
    }

    else {
      printf("Buffer full: Received data not saved\n");
    }
  //}
  return slot;
}

extern "C" int cSend(int commIdex, int name) {
  //char message[1024] = "Server Received";
  int nm = name;
  send(comm_fd[commIdex][0], &nm, sizeof(int), 0);
  return 0;
}

extern "C" int cCloseListen() {;
  close(listen_fd);
  return 0;
}

extern "C" int cCloseComm(int commIndex) {
  close(comm_fd[commIndex][0]);
  comm_fd[commIndex][1] = SLOTEMPTY;
  printf("Closed Comm\n");
  return 0;
}



//get functions
extern "C" int getID(int slot) {
  return (recvTable[slot][0]);
}

//get functions
extern "C" int getCamera(int slot) {
  return (recvTable[slot][1]);
}

//get functions
extern "C" int getXPos(int slot) {
  return (recvTable[slot][2]);
}

//get functions
extern "C" int getYPos(int slot) {
  return (recvTable[slot][3]);
}

//get functions
extern "C" int getHeight(int slot) {
  return (recvTable[slot][4]);
}

//get functions
extern "C" int getWidth(int slot) {
  return (recvTable[slot][5]);
}

//clear function
extern "C" int clearSlot(int slot) {
  recvTable[slot][NUMSTRUCTELEMENTS] = SLOTEMPTY;
  return 0;
}


//print local table
extern "C" void printTable() {
  for(int i = 0; i < TABLESIZE; ++i) {
    printf("Row: %d name: %d camera: %d cx: %d cy: %d h: %d w: %d b: %d\n", i,
      recvTable[i][0], recvTable[i][1], recvTable[i][2], recvTable[i][3],
      recvTable[i][4], recvTable[i][5], recvTable[i][6]);
  }
}

/*
getCamera
getID
getXPos
getYPos
get
*/








/*int main(int argc, char const *argv[]) 
{ 
  int status = server();

  return 0;
} */

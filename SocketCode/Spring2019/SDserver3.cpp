#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include <time.h>

#include <unistd.h>
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <string.h> 
#include <arpa/inet.h>

#include <queue>
#include <pthread.h>

#define rPORT 76543
#define sPORT 76544
#define lIPADDR INADDR_ANY
#define NUM_THREADS 2
#define NUM_NODES 4
using namespace std;

//g++ SDserver.cpp -pthread -o SDserver

pthread_mutex_t sendLocks[NUM_NODES];
pthread_mutex_t recvLocks[NUM_NODES];
pthread_mutex_t statusLocks[NUM_NODES];
int nodeStatus[NUM_NODES] = {0};
int comm_fd[NUM_NODES];
char* clientIP[NUM_NODES];
  
vector<queue <int> > sQList;
vector<queue <int> > rQList;

//queue<int> sendQ;
//queue<int> recvQ;

int sendLock = 0;
int recvLock = 1;




void *sendData(void* index) {
   int statIndex = *((int*) index);
   int status = 0;
          cout << "send" << endl;
   while(1) {
    /* Initializing network parameters */  
    pthread_mutex_lock(&statusLocks[statIndex]);
    status = nodeStatus[statIndex];
    pthread_mutex_unlock(&statusLocks[statIndex]);
    if(status > 0) {
    std::cout << "send status: " << status << std::endl;
      int sock;
      pthread_mutex_lock(&statusLocks[statIndex]);
      char* IPADDR = clientIP[statIndex];
      pthread_mutex_unlock(&statusLocks[statIndex]);
	    struct sockaddr_in servaddr;

	    sock = socket(AF_INET,SOCK_STREAM,0);
	    bzero(&servaddr,sizeof servaddr);

	    servaddr.sin_family=AF_INET;
	    servaddr.sin_port=htons(sPORT);

	    inet_pton(AF_INET, IPADDR, &(servaddr.sin_addr));

	    connect(sock,(struct sockaddr *)&servaddr,sizeof(servaddr)); 

      int data = 5;

      while(status > 0) {
        pthread_mutex_lock(&sendLocks[statIndex]);
        if(!sQList[statIndex].empty()) {
          data = sQList[statIndex].front();
          sQList[statIndex].pop();
          pthread_mutex_unlock(&sendLocks[statIndex]);
          int retCode = send(sock, &data, sizeof(int), 0);
          std::cout << "retCode: " << retCode << std::endl;
          pthread_mutex_lock(&statusLocks[statIndex]);
          status = nodeStatus[statIndex];
          pthread_mutex_unlock(&statusLocks[statIndex]);
        }
        pthread_mutex_unlock(&sendLocks[statIndex]);
      }
      close(sock);
    }
  }
}


void *recvData(void* index) {
  int statIndex = *((int*) index);
  int data = 0;
  int retCode = 1;
          cout << "recv" << endl;
  while(1) {
    pthread_mutex_lock(&statusLocks[statIndex]);
    int status = nodeStatus[statIndex];
    pthread_mutex_unlock(&statusLocks[statIndex]);

    if(status > 0)  {
      while(retCode > 0) {
        retCode = recv(comm_fd[statIndex], &data, sizeof(int), 0);
        
        pthread_mutex_lock(&recvLocks[statIndex]);
        rQList[statIndex].push(data);
        pthread_mutex_unlock(&recvLocks[statIndex]);
    
        cout << data << " status: " << status << endl;
      }
      pthread_mutex_lock(&statusLocks[statIndex]);
      nodeStatus[statIndex] = 0;
      pthread_mutex_unlock(&statusLocks[statIndex]);
      close(comm_fd[statIndex]);
    }
  }
}

void *listenFunc(void*) {
  pthread_t recvThreads[NUM_NODES];
  pthread_t sendThreads[NUM_NODES];

  int statusIndex[NUM_NODES];

   int listen_fd;

	  struct sockaddr_in servaddr;
    struct sockaddr_in clientAddr;
    socklen_t clLen = sizeof(clientAddr);

	  listen_fd = socket(AF_INET, SOCK_STREAM, 0);

	  bzero( &servaddr, sizeof(servaddr));

	  servaddr.sin_family = AF_INET;
	  servaddr.sin_addr.s_addr = htons(lIPADDR);
	  servaddr.sin_port = htons(rPORT);

	  bind(listen_fd, (struct sockaddr *) &servaddr, sizeof(servaddr));

  for(int i = 0; i < NUM_NODES; ++i) {
    statusIndex[i] = i;
    clientIP[i] = NULL;
    pthread_create(&recvThreads[i], NULL, recvData, (void*) &statusIndex[i]);
    pthread_create(&sendThreads[i], NULL, sendData, (void*) &statusIndex[i]);
    cout << i << endl;
  }

  while(1) {
    listen(listen_fd, 10);
    int comm_Temp = accept(listen_fd, (struct sockaddr*) &clientAddr, &clLen);
    int useIndex = -1;
    for (int i = 0; i < NUM_NODES; ++i) {
      if(clientIP[i] == NULL) {
        useIndex = i;
      }
      else {
        if(!strcmp(clientIP[i], inet_ntoa(clientAddr.sin_addr))) {
          useIndex = i;
          i = NUM_NODES;
        }
      }
    }

    if(useIndex > -1) {
      pthread_mutex_lock(&statusLocks[useIndex]);
      clientIP[useIndex] = inet_ntoa(clientAddr.sin_addr);
      comm_fd[useIndex] = comm_Temp;
      nodeStatus[useIndex] = 1;
      pthread_mutex_unlock(&statusLocks[useIndex]);
    }
  }
  close(listen_fd);

  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(recvThreads[i], NULL);
    pthread_join(sendThreads[i], NULL);
  }

}

//g++ SDclient3.cpp -o SDclient3

int main(int argc, char const *argv[]) {

  for(int i = 0; i < NUM_NODES; ++i) {
    sQList.push_back(queue<int>());
    rQList.push_back(queue<int>());

    pthread_mutex_init(&sendLocks[i], NULL);
    pthread_mutex_init(&recvLocks[i], NULL);
    pthread_mutex_init(&statusLocks[i], NULL);
  }

  pthread_t listenThread;
  pthread_create(&listenThread, NULL, listenFunc, NULL);

  sleep(3);
  int j = 0;
  while(1) {

  for(int i = 0; i < NUM_NODES; ++i) {
    pthread_mutex_lock(&sendLocks[i]);
    sQList[i].push(j);
    pthread_mutex_unlock(&sendLocks[i]);
  }

  for(int i = 0; i < NUM_NODES; ++i) {
    pthread_mutex_lock(&recvLocks[i]);
    if(!rQList[i].empty()) {    
      rQList[i].pop();
    }
    pthread_mutex_unlock(&recvLocks[i]);
  }
    ++j;
  if(j%30==0) {
    sleep(2);
  }  
 }



  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(listenThread, NULL);
  }


  return 0;
}

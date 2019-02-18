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
string clientIP[NUM_NODES];

vector<queue <int> > sQList;
vector<queue <int> > rQList;



/* Function: sendData
    Use: Send data to edge node in independant thread
    Input:
      index: index of connedtion for stored socket and send table
*/
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
      /* Initilize send socket*/
      int sock;
      pthread_mutex_lock(&statusLocks[statIndex]);
      const char* IPADDR = clientIP[statIndex].c_str();
      pthread_mutex_unlock(&statusLocks[statIndex]);
	    struct sockaddr_in servaddr;

	    sock = socket(AF_INET,SOCK_STREAM,0);
	    bzero(&servaddr,sizeof servaddr);

	    servaddr.sin_family=AF_INET;
	    servaddr.sin_port=htons(sPORT);

	    inet_pton(AF_INET, IPADDR, &(servaddr.sin_addr));

	    connect(sock,(struct sockaddr *)&servaddr,sizeof(servaddr)); 

      int data = 5;
      /* While the connection is still valid*/
      while(status > 0) {
        pthread_mutex_lock(&sendLocks[statIndex]);
        if(!sQList[statIndex].empty()) {
          /* Read value from send queue and send it */
          data = sQList[statIndex].front();
          sQList[statIndex].pop();
          pthread_mutex_unlock(&sendLocks[statIndex]);
          int retCode = send(sock, &data, sizeof(int), 0);
          std::cout << "retCode: " << retCode << std::endl;
          /* Check status to make sure connection is still valid*/
          pthread_mutex_lock(&statusLocks[statIndex]);
          status = nodeStatus[statIndex];
          pthread_mutex_unlock(&statusLocks[statIndex]);
        }
        pthread_mutex_unlock(&sendLocks[statIndex]);
      }
      close(sock);
      cout << "send sock closed" << endl;
    }
  }
}

/* Function: recData
    Use: Recv data from edge node in independant thread
    Input:
      index: index of connedtion for stored socket and send table
*/
void *recvData(void* index) {
  int statIndex = *((int*) index);
  int data = 0;
          cout << "recv" << endl;
  while(1) {
    /* Wait for status to be 1 before running */
    pthread_mutex_lock(&statusLocks[statIndex]);
    int status = nodeStatus[statIndex];
    pthread_mutex_unlock(&statusLocks[statIndex]);

    if(status > 0)  {
      /* return code for recv function */
      int retCode = 1;
      while(retCode > 0) {
        retCode = recv(comm_fd[statIndex], &data, sizeof(int), 0);
        /* Lock and push data into queue */
        pthread_mutex_lock(&recvLocks[statIndex]);
        rQList[statIndex].push(data);
        pthread_mutex_unlock(&recvLocks[statIndex]);
    
        cout << data << " recv retCode: " << retCode << endl;
      }
      /*When disconnected, close socket*/
      pthread_mutex_lock(&statusLocks[statIndex]);
      nodeStatus[statIndex] = 0;
      pthread_mutex_unlock(&statusLocks[statIndex]);
      close(comm_fd[statIndex]);
      cout << "recv sock closed" << endl;
    }
  }
}


/* Function: listenFunc
    Use: Listen and accept connections in independant thread
*/
void *listenFunc(void*) {

  /* Initilize Thread IDs for send and recv*/
  pthread_t recvThreads[NUM_NODES];
  pthread_t sendThreads[NUM_NODES];

  /* Index used for thread organization*/
  int statusIndex[NUM_NODES];

  /* Initilize socket for listening*/
   int listen_fd;

    /* Initilize server: General structure*/
    struct sockaddr_in servaddr;
    struct sockaddr_in clientAddr;
    socklen_t clLen = sizeof(clientAddr);

	  listen_fd = socket(AF_INET, SOCK_STREAM, 0);

	  bzero( &servaddr, sizeof(servaddr));

	  servaddr.sin_family = AF_INET;
	  servaddr.sin_addr.s_addr = htons(lIPADDR);
	  servaddr.sin_port = htons(rPORT);

	  bind(listen_fd, (struct sockaddr *) &servaddr, sizeof(servaddr));

  /* Spawn send and recv threads*/
  for(int i = 0; i < NUM_NODES; ++i) {
    statusIndex[i] = i;
    clientIP[i].clear();
    pthread_create(&recvThreads[i], NULL, recvData, (void*) &statusIndex[i]);
    pthread_create(&sendThreads[i], NULL, sendData, (void*) &statusIndex[i]);
    cout << i << endl;
  }

  while(1) {
    /* Listen for new connection and accept*/
    listen(listen_fd, 10);
    /* Store the IP address of Client in clientAddr*/
    int comm_Temp = accept(listen_fd, (struct sockaddr*) &clientAddr, &clLen);
    cout << "IPnew: " << inet_ntoa(clientAddr.sin_addr) << endl;
    int useIndex = -1;
    for (int i = 0; i < NUM_NODES; ++i) {
      if(clientIP[i].empty()) {
        useIndex = i;
      }
      else {
        if(strcmp(clientIP[i].c_str(), inet_ntoa(clientAddr.sin_addr))==0) {
          //cout << "IPnew2: " << inet_ntoa(clientAddr.sin_addr) << endl;
          //cout << "IPold: " << clientIP[i] << endl;
          useIndex = i;
          i = NUM_NODES;
          //cout << "found index: " << useIndex << endl;
        }
      }
    }
    cout << "UseIndex: " << useIndex << endl;
    if(useIndex > -1) {
      pthread_mutex_lock(&statusLocks[useIndex]);
if(!clientIP[useIndex].empty()) {
      cout << "IPb: " << clientIP[useIndex] << endl;
}
      string str(inet_ntoa(clientAddr.sin_addr));
      clientIP[useIndex] = str;
      cout << "IPa: " << clientIP[useIndex] << endl;
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

/* Function: main
    Use: Initilize listen thread and handle test data
*/
int main(int argc, char const *argv[]) {

  for(int i = 0; i < NUM_NODES; ++i) {
    /* Intialize Vector of Queues for send and recieve data*/
    sQList.push_back(queue<int>());
    rQList.push_back(queue<int>());
    /* Initializing all Mutexes */
    pthread_mutex_init(&sendLocks[i], NULL);
    pthread_mutex_init(&recvLocks[i], NULL);
    pthread_mutex_init(&statusLocks[i], NULL);
  }

  /* Create thread ID for listen Thread*/
  pthread_t listenThread;
  /* Launch thread to listen for incoming connections*/
  pthread_create(&listenThread, NULL, listenFunc, NULL);

  sleep(3);
  int j = 0;
  while(1) {

    /* Push values to send into send queue*/
    for(int i = 0; i < NUM_NODES; ++i) {
      pthread_mutex_lock(&sendLocks[i]);
      sQList[i].push(j);
      pthread_mutex_unlock(&sendLocks[i]);
    }

    /* Pop values out of recv queue*/
    for(int i = 0; i < NUM_NODES; ++i) {
      pthread_mutex_lock(&recvLocks[i]);
      if(!rQList[i].empty()) {    
        rQList[i].pop();
      }
      pthread_mutex_unlock(&recvLocks[i]);
    }
      ++j;
    if(j%30==0) {
      sleep(3);
    }  
 }



  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(listenThread, NULL);
  }


  return 0;
}

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

#define sPORT 76543
#define rPORT 76544
#define lIPADDR INADDR_ANY
#define NUM_THREADS 2

using namespace std;

pthread_mutex_t locks[NUM_THREADS];
  
queue<int> sendQ;
queue<int> recvQ;

int sendLock = 0;
int recvLock = 1;


void *sendData(void*) {
  /* Initializing network parameters */
  int sock;
  char* IPADDR = "127.0.0.1";
	struct sockaddr_in servaddr;

	sock = socket(AF_INET,SOCK_STREAM,0);
	bzero(&servaddr,sizeof servaddr);

	servaddr.sin_family=AF_INET;
	servaddr.sin_port=htons(sPORT);

	inet_pton(AF_INET, IPADDR, &(servaddr.sin_addr));

	connect(sock,(struct sockaddr *)&servaddr,sizeof(servaddr)); 

  int data = 5;

  while(1) {
    pthread_mutex_lock(&locks[sendLock]);
    if(!sendQ.empty()) {
      data = sendQ.front();
      sendQ.pop();
      pthread_mutex_unlock(&locks[sendLock]);
      int status = send(sock, &data, sizeof(int), 0);
    }
  }
  close(sock);
}


void *recvData(void*) {

  while(1) {
   int listen_fd, comm_fd;

	  struct sockaddr_in servaddr;

	  listen_fd = socket(AF_INET, SOCK_STREAM, 0);

	  bzero( &servaddr, sizeof(servaddr));

	  servaddr.sin_family = AF_INET;
	  servaddr.sin_addr.s_addr = htons(lIPADDR);
	  servaddr.sin_port = htons(rPORT);

	  bind(listen_fd, (struct sockaddr *) &servaddr, sizeof(servaddr));

    listen(listen_fd, 10);
    comm_fd = accept(listen_fd, (struct sockaddr*) NULL, NULL);

    int data = 0;
    int status = 1;

    while(status > 0) {
      status = recv(comm_fd, &data, sizeof(int), 0);
      
      pthread_mutex_lock(&locks[recvLock]);
      recvQ.push(data);
      pthread_mutex_unlock(&locks[recvLock]);

      cout << data << " status: " << status << endl;
    }
    close(comm_fd);
    close(listen_fd);
  }

}

//g++ SDclient3.cpp -pthread -o SDclient3

int main(int argc, char const *argv[]) {
  pthread_t threads[NUM_THREADS];

  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_mutex_init(&locks[i], NULL);
  }

  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_create(&threads[i], NULL, sendData, NULL);
    ++i;
    pthread_create(&threads[i], NULL, recvData, NULL);
  }


  int j = 0;
  while(j < 30) {
    pthread_mutex_lock(&locks[sendLock]);
    sendQ.push(j);
    pthread_mutex_unlock(&locks[sendLock]);

    pthread_mutex_lock(&locks[recvLock]);
    if(!recvQ.empty()) {    
      recvQ.pop();
    }
    pthread_mutex_unlock(&locks[recvLock]);
    ++j;  
  }



  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }


  return 0;
}

#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include <time.h>

#include <unistd.h>
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <string.h> 
#include <arpa/inet.h>
#include <vector>
#include <pthread.h>

#define NUM_THREADS 30
#define PORT 65432

//g++ sockettest.cpp -o client `pkg-config --cflags --libs opencv`
//g++ -pthread clientT.cpp -o client `pkg-config --cflags --libs opencv`

//#include "opencv2/opencv.hpp"
//using namespace cv;

using namespace std;

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
  int localIndex;
  //Mat hist;
};

#define TABLESIZE   30
struct uniqueID table[TABLESIZE];
pthread_mutex_t locks[TABLESIZE];


void *thread(void* arg) {
  int sock;
  char* IPADDR = "127.0.0.1";
	//char recvline[1024];
  int newName = -1;
	struct sockaddr_in servaddr;

  uniqueID* testStruct = (uniqueID*) arg;


	sock = socket(AF_INET,SOCK_STREAM,0);
	bzero(&servaddr,sizeof servaddr);

	servaddr.sin_family=AF_INET;
	servaddr.sin_port=htons(PORT);

	inet_pton(AF_INET, IPADDR, &(servaddr.sin_addr));

	connect(sock,(struct sockaddr *)&servaddr,sizeof(servaddr)); 

  int status = 0;
  int retcode = -1;
    status = send(sock, &testStruct->uInfo, sizeof((testStruct->uInfo)) , 0 );
    //printf("Hello messages sent\n"); 
    retcode = recv( sock , &newName, sizeof(int), 0); 
    //printf("Received: %d retcode: %d\n", newName, retcode); 
    if(retcode>0) {
      pthread_mutex_unlock(&locks[testStruct->localIndex]);
      table[testStruct->localIndex].uInfo.name = newName;
      pthread_mutex_unlock(&locks[testStruct->localIndex]);
    }

  close(sock);
}



int main() {

srand (time(NULL));
int counter = 0;
string cameraID = "A";


pthread_t threads[TABLESIZE];
/* initialize mutex set (non-recursive lock) */
for (int i = 0; i < TABLESIZE; i++)
 pthread_mutex_init(&locks[i], NULL);
int result_code = -1;


//Populate table and send each entry to server
for (int i=0; i<TABLESIZE; i++) {
  pthread_mutex_lock(&locks[i]);
	table[i].uInfo.name = i;
	table[i].uInfo.cx = rand() % 200 + 1;
  table[i].localIndex = i;
	printf("Name: %d UniqueID: %d \n", table[i].uInfo.name, table[i].uInfo.cx) ;
	//send to server
    //printf("IN MAIN: Creating thread %d.\n", i);
    threads[i] = i;
    result_code = pthread_create(&threads[i], NULL, thread, (void*) &table[i]);
    pthread_mutex_unlock(&locks[i]);
    //sleep(2);
}

  //printf("IN MAIN: All threads are created.\n");


  //wait for each thread to complete
  for (int i = 0; i < TABLESIZE; i++) {
    result_code = pthread_join(threads[i], NULL);
    //assert(!result_code);
   // printf("IN MAIN: Thread %d has ended.\n", i);
  }

  //printf("MAIN program has ended.\n");

/***************************************/



// Resend same table w/ indices reversed
for (int i = TABLESIZE-1; i >= 0; i--) {
  pthread_mutex_lock(&locks[i]);
	table[i].uInfo.name = TABLESIZE-1-i;
	//printf("Name: %d UniqueID: %d \n", table[i].uInfo.name, table[i].uInfo.cx) ;
	//send to server
  //printf("IN MAIN: Creating thread %d.\n", i);
  threads[i] = i;
  result_code = pthread_create(&threads[i], NULL, thread, (void*) &table[i].uInfo);
  pthread_mutex_unlock(&locks[i]);
  //sleep(2);
	//recieve new name
	//update new name
	//table[i].name = recieved_name;
}

  //printf("IN MAIN: All threads are created.\n");

  //wait for each thread to complete
  for (int i = 0; i < TABLESIZE; i++) {
    result_code = pthread_join(threads[i], NULL);
    //assert(!result_code);
    //printf("IN MAIN: Thread %d has ended.\n", i);
  }

 // printf("MAIN program has ended.\n");
for (int i = 0; i < TABLESIZE; i++) {
    printf("Name: %d UniqueID: %d \n", table[i].uInfo.name, table[i].uInfo.cx);
  }




return 0;
}

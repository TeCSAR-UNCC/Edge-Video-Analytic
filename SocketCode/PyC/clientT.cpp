#include <unistd.h> 
#include <stdio.h> 
#include <sys/socket.h> 
#include <stdlib.h> 
#include <netinet/in.h> 
#include <string.h> 
#include <arpa/inet.h>
#include <vector>
#include <pthread.h>

#define NUM_THREADS 5
#define PORT 65432
//g++ sockettest.cpp -o client `pkg-config --cflags --libs opencv`
//g++ -pthread clientT.cpp -o client `pkg-config --cflags --libs opencv`

#include "opencv2/opencv.hpp"
using namespace cv;

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
  Mat hist;
  //uint8_t hits;
  //time
};

void testStructCreate(Info &testStruct, int num);


void *thread(void* arg) {
  int sock;
  char* IPADDR = "127.0.0.1";
	char recvline[1024];
	struct sockaddr_in servaddr;

  Info* testStruct = (Info*) arg;


	sock = socket(AF_INET,SOCK_STREAM,0);
	bzero(&servaddr,sizeof servaddr);

	servaddr.sin_family=AF_INET;
	servaddr.sin_port=htons(PORT);

	inet_pton(AF_INET, IPADDR, &(servaddr.sin_addr));

	connect(sock,(struct sockaddr *)&servaddr,sizeof(servaddr)); 

  int status = 0;
    status = send(sock, testStruct, sizeof(*testStruct) , 0 );
    printf("Hello messages sent\n"); 
    recv( sock , &recvline, sizeof(recvline), 0); 
    printf("Received: %s\n", recvline); 


  close(sock);
  return 0;
}


int main(int argc, char const *argv[]) 
{ 

  pthread_t threads[NUM_THREADS];
  Info testStruct[NUM_THREADS];
  int result_code = -1;


    //create all threads one by one
  for (int i = 0; i < NUM_THREADS; i++) {
    printf("IN MAIN: Creating thread %d.\n", i);
    threads[i] = i;
    testStructCreate(testStruct[i], i);
    printf("%d\n",testStruct[i].cx);
    result_code = pthread_create(&threads[i], NULL, thread, (void*) &testStruct[i]);
    assert(!result_code);
  }

  printf("IN MAIN: All threads are created.\n");

  //wait for each thread to complete
  for (int i = 0; i < NUM_THREADS; i++) {
    result_code = pthread_join(threads[i], NULL);
    assert(!result_code);
    printf("IN MAIN: Thread %d has ended.\n", i);
  }

  printf("MAIN program has ended.\n");

  
  return 0; 
} 




void testStructCreate(Info &testStruct, int num) {

  if(num == 0) {
    testStruct.cx = 0;
    testStruct.cy = 0;
    testStruct.h = 0;
    testStruct.w = 0;
    testStruct.name = 0;
    testStruct.cameraID = 0;
  }
  else if(num == 1) {
    testStruct.cx = 1;
    testStruct.cy = 1;
    testStruct.h = 1;
    testStruct.w = 1;
    testStruct.name = 1;
    testStruct.cameraID = 0;
  }

  else if(num == 3) {
  testStruct.cx = 0;
    testStruct.cy = 0;
    testStruct.h = 0;
    testStruct.w = 0;
    testStruct.name = 4;
    testStruct.cameraID = 1;
  }

  else {
    testStruct.cx = 1;
    testStruct.cy = 1;
    testStruct.h = 1;
    testStruct.w = 1;
    testStruct.name = 5;
    testStruct.cameraID = 1;
  }
}

#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include <time.h>

#include <unistd.h>
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <string.h> 
#include <arpa/inet.h>

#include <opencv2/opencv.hpp>

#include <queue>
#include <pthread.h>

#define rPORT 76543
#define sPORT 76544
#define lIPADDR INADDR_ANY
#define NUM_THREADS 2
#define NUM_NODES 2
#define SIZEOFDB 1000
#define OUTPUT_SIZE 1280
#define EUC_THRESH 4.25   //4.25 mobilenet 
using namespace std;

//g++ SDserver.cpp -pthread -o SDserver

pthread_mutex_t sendLocks[NUM_NODES];
pthread_mutex_t recvLocks[NUM_NODES];
pthread_mutex_t statusLocks[NUM_NODES];
int nodeStatus[NUM_NODES] = {0};
int comm_fd[NUM_NODES];
string clientIP[NUM_NODES];


struct personType
{
	int currentCamera;
	int label;
	float fv_array[OUTPUT_SIZE];
	float xPos;
	float yPos;
	float height;
	float width;

};

struct object_history
{
	personType personObject;
	cv::Mat fv;
	int lru;
};

struct reIDType
{
	int oldID;
	int newID;
	//int newCameraID;
};

vector<queue <reIDType> > sQList;
vector<queue <personType> > rQList;

object_history dataBase[SIZEOFDB];

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

      reIDType data;
      /* While the connection is still valid*/
      while(status > 0) {
        pthread_mutex_lock(&sendLocks[statIndex]);
        if(!sQList[statIndex].empty()) {
          /* Read value from send queue and send it */
          data = sQList[statIndex].front();
          sQList[statIndex].pop();
          pthread_mutex_unlock(&sendLocks[statIndex]);
          int retCode = send(sock, &data, sizeof(reIDType), 0);
          //std::cout << "retCode: " << retCode << std::endl;
          /* Check status to make sure connection is still valid*/
        }
        pthread_mutex_lock(&statusLocks[statIndex]);
        status = nodeStatus[statIndex];
        pthread_mutex_unlock(&statusLocks[statIndex]);
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
  personType data;
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

       	retCode = recv(comm_fd[statIndex], reinterpret_cast<char*>(&data), sizeof(personType), MSG_WAITALL);
        cout << "data.label = " << data.label << endl;
        /* Lock and push data into queue */
		    cout << "retCode: " << retCode << endl;
        //retCode = recv(comm_fd[statIndex], &data, sizeof(personType), 0);
        pthread_mutex_lock(&recvLocks[statIndex]);
        rQList[statIndex].push(data);
        pthread_mutex_unlock(&recvLocks[statIndex]);
    
        //cout << data << " recv retCode: " << retCode << endl;
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

  // Initilize Thread IDs for send and recv
  pthread_t recvThreads[NUM_NODES];
  pthread_t sendThreads[NUM_NODES];

  // Index used for thread organization
  int statusIndex[NUM_NODES];

  // Initilize socket for listening
   int listen_fd;

    // Initilize server: General structure
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


int main(int argc, char const *argv[]) {
	int currentIndex = 0;

  for(int i=0; i < SIZEOFDB; i++) {
    dataBase[i].personObject.label = -1;
  }

  for(int i = 0; i < NUM_NODES; ++i) {
    // Intialize Vector of Queues for send and recieve data
    sQList.push_back(queue<reIDType>());
    rQList.push_back(queue<personType>());
   
    // Initializing all Mutexes 
    pthread_mutex_init(&sendLocks[i], NULL);
    pthread_mutex_init(&recvLocks[i], NULL);
    pthread_mutex_init(&statusLocks[i], NULL);
  }

  // Create thread ID for listen Thread
  pthread_t listenThread;
  // Launch thread to listen for incoming connections
  pthread_create(&listenThread, NULL, listenFunc, NULL);

  sleep(3);
  int j = 0;
  reIDType pushData;
	personType tmpPerson;
	
  while(1) {
    // Pop values out of recv queue
    for(int i = 0; i < NUM_NODES; ++i) {
    	int matchIndex = -1;
			int tmpMatchIndex = -1;
			float tmpEucDist = 1000;
			float minEucDist = 1000;
			int max = -1;
			int useIndex = -1;
			bool updateFlag = false;
      pthread_mutex_lock(&recvLocks[i]);
      if(!rQList[i].empty()) { 
				tmpPerson = rQList[i].front();
				cout << "tmpPerson.label = " << tmpPerson.label << endl;
				rQList[i].pop();
				pthread_mutex_unlock(&recvLocks[i]);
				
				//Get mat for L2norm comparision
				cv::Mat matTP(1,OUTPUT_SIZE, CV_32F);  
				void* ptr = &tmpPerson.fv_array;
				std::memcpy(matTP.data, ptr, OUTPUT_SIZE*sizeof(float));
				
				//Do comparisone with dataBase
				for(int j=0; j < SIZEOFDB; j++){
					if(dataBase[j].personObject.label > -1) {
						dataBase[j].lru++;

            if(tmpPerson.currentCamera > -1) { //if not an unlock recv
						  tmpEucDist = cv::norm(dataBase[j].fv, matTP, cv::NORM_L2);
              if( (dataBase[j].personObject.label == tmpPerson.label) && (dataBase[j].personObject.currentCamera == tmpPerson.currentCamera) ) {
							  updateFlag = true;
							  matchIndex = j;
							  j = SIZEOFDB;
              }
						  else if( (tmpEucDist < minEucDist) && (dataBase[j].personObject.currentCamera == -1) ) {
							  minEucDist = tmpEucDist;
							  tmpMatchIndex = j;
						  }
					  }
            else { //unlock
              if(dataBase[j].personObject.label == tmpPerson.label) {
                dataBase[j].personObject.currentCamera = -1;
                j = SIZEOFDB;
              }
            }
          }
				}

        if(tmpPerson.currentCamera > -1) {
				  if( (minEucDist < EUC_THRESH) && (updateFlag == false) ) {
					  matchIndex = tmpMatchIndex;
				  }	
				  if(matchIndex >= 0){
					  cout << "MatchIndex = " << matchIndex << endl;
					  dataBase[matchIndex].personObject.xPos = tmpPerson.xPos;
					  dataBase[matchIndex].personObject.yPos = tmpPerson.yPos;
					  dataBase[matchIndex].personObject.height = tmpPerson.height;
					  dataBase[matchIndex].personObject.width = tmpPerson.width;
            dataBase[matchIndex].personObject.currentCamera = tmpPerson.currentCamera;
					  memcpy(&dataBase[matchIndex].personObject.fv_array, &tmpPerson.fv_array, OUTPUT_SIZE*sizeof(float));
					  matTP.copyTo(dataBase[matchIndex].fv);
					  //cout << "fv_array[0] = " << dataBase[matchIndex].personObject.fv_array[0] << endl;
					  dataBase[matchIndex].lru = 0;
					  pushData.oldID = tmpPerson.label;
					  pushData.newID = dataBase[matchIndex].personObject.label;
					  //pushData.newCameraID = dataBase[matchIndex].personObject.cameraID;
					  cout << "pushData.newID = " << pushData.newID << " and tmpPerson.label = " << tmpPerson.label << " and dataBase[matchIndex].personObject.label = " << dataBase[matchIndex].personObject.label << endl;
					  pthread_mutex_lock(&sendLocks[i]);
					  sQList[i].push(pushData);
					  pthread_mutex_unlock(&sendLocks[i]);
				  }
				  else if (currentIndex < SIZEOFDB){
					  cout << "currentIndex = " << currentIndex << endl;
					  dataBase[currentIndex].personObject.xPos = tmpPerson.xPos;
					  dataBase[currentIndex].personObject.yPos = tmpPerson.yPos;
					  dataBase[currentIndex].personObject.height = tmpPerson.height;
					  dataBase[currentIndex].personObject.width = tmpPerson.width;
            dataBase[currentIndex].personObject.currentCamera = tmpPerson.currentCamera;
					  dataBase[currentIndex].personObject.label = tmpPerson.label;
					  memcpy(&dataBase[currentIndex].personObject.fv_array, &tmpPerson.fv_array, OUTPUT_SIZE*sizeof(float));
					  matTP.copyTo(dataBase[currentIndex].fv);
					  pushData.oldID = tmpPerson.label;
					  pushData.newID = dataBase[currentIndex].personObject.label;
					  //pushData.newCameraID = dataBase[matchIndex].personObject.cameraID;
					  cout << "pushData.newID = " << pushData.newID << " and tmpPerson.label = " << tmpPerson.label << " and dataBase[matchIndex].personObject.label = " << dataBase[currentIndex].personObject.label << endl;
					  pthread_mutex_lock(&sendLocks[i]);
					  sQList[i].push(pushData);
					  pthread_mutex_unlock(&sendLocks[i]);
					  dataBase[currentIndex].lru = 0;
					  currentIndex++;
				  }
				  else{
					  cout << "Database is full. Initiating replacement policy..." << endl;
					  for(int k=0; k < SIZEOFDB; k++){
						  if(max < dataBase[k].lru){
							  max = dataBase[k].lru;
							  useIndex = k;
						  }
					  }
					  dataBase[useIndex].personObject.xPos = tmpPerson.xPos;
					  dataBase[useIndex].personObject.yPos = tmpPerson.yPos;
					  dataBase[useIndex].personObject.height = tmpPerson.height;
					  dataBase[useIndex].personObject.width = tmpPerson.width;
            dataBase[useIndex].personObject.currentCamera = tmpPerson.currentCamera;
					  dataBase[useIndex].personObject.label = tmpPerson.label;
					  memcpy(&dataBase[useIndex].personObject.fv_array, &tmpPerson.fv_array, OUTPUT_SIZE*sizeof(float));
					  matTP.copyTo(dataBase[useIndex].fv);
					  pushData.oldID = tmpPerson.label;
					  pushData.newID = dataBase[useIndex].personObject.label;
					  //pushData.newCameraID = dataBase[matchIndex].personObject.cameraID;
					  cout << "pushData.newID = " << pushData.newID << " and tmpPerson.label = " << tmpPerson.label << " and dataBase[matchIndex].personObject.label = " << dataBase[useIndex].personObject.label << endl;
					  pthread_mutex_lock(&sendLocks[i]);
					  sQList[i].push(pushData);
					  pthread_mutex_unlock(&sendLocks[i]);
					  dataBase[useIndex].lru = 0;
				  }
        }
			}	
			else{
				pthread_mutex_unlock(&recvLocks[i]);	
			}	
		}  
	}

  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(listenThread, NULL);
  }


  return 0;
}


/*
int main(int argc, char const *argv[]) {

  for(int i=0; i < SIZEOFDB; i++) {
    dataBase[i].personObject.cameraID = -1;
  }

	int currentIndex = 0;
  for(int i = 0; i < NUM_NODES; ++i) {
    // Intialize Vector of Queues for send and recieve data
    sQList.push_back(queue<reIDType>());
    rQList.push_back(queue<personType>());
   
    //setsockopt(comm_fd[i], SOL_SOCKET, SO_RCVBUF, &buffersize, sizeof(buffersize));
    // Initializing all Mutexes 
    pthread_mutex_init(&sendLocks[i], NULL);
    pthread_mutex_init(&recvLocks[i], NULL);
    pthread_mutex_init(&statusLocks[i], NULL);
  }

  // Create thread ID for listen Thread
  pthread_t listenThread;
  // Launch thread to listen for incoming connections
  pthread_create(&listenThread, NULL, listenFunc, NULL);

  sleep(3);
  int j = 0;
  reIDType pushData;
	personType tmpPerson;
	
  while(1) {
		int matchIndex = -1;
    // Pop values out of recv queue
    for(int i = 0; i < NUM_NODES; ++i) {
      pthread_mutex_lock(&recvLocks[i]);
      if(!rQList[i].empty()) { 
				tmpPerson = rQList[i].front();
				cout << "tmpPerson.label = " << tmpPerson.label << endl;
				rQList[i].pop();
				pthread_mutex_unlock(&recvLocks[i]);
				//Do comparisone with dataBase
				for(int j=0; j < SIZEOFDB; j++){
					if(dataBase[j].personObject.cameraID > -1) {
						if(dataBase[j].personObject.label == tmpPerson.label) {
							matchIndex = j;
							j = SIZEOFDB;
						}
					}
				}	
				if(matchIndex >= 0){
					cout << "MatchIndex = " << matchIndex << endl;
					dataBase[matchIndex].personObject.xPos = tmpPerson.xPos;
					dataBase[matchIndex].personObject.yPos = tmpPerson.yPos;
					dataBase[matchIndex].personObject.height = tmpPerson.height;
					dataBase[matchIndex].personObject.width = tmpPerson.width;
					memcpy(&dataBase[matchIndex].personObject.fv_array, &tmpPerson.fv_array, OUTPUT_SIZE*sizeof(float));
					cout << "fv_array[0] = " << dataBase[matchIndex].personObject.fv_array[0] << endl;
					dataBase[matchIndex].lru = 0;
					pushData.oldID = tmpPerson.label;
					pushData.newID = dataBase[matchIndex].personObject.label;
					pushData.newCameraID = dataBase[matchIndex].personObject.cameraID;
					cout << "pushData.newID = " << pushData.newID << " and tmpPerson.label = " << tmpPerson.label << " and dataBase[matchIndex].personObject.label = " << dataBase[matchIndex].personObject.label << endl;
					pthread_mutex_lock(&sendLocks[i]);
					sQList[i].push(pushData);
					pthread_mutex_unlock(&sendLocks[i]);
				}
				else if (currentIndex < SIZEOFDB){
					cout << "currentIndex = " << currentIndex << endl;
					dataBase[currentIndex].personObject.xPos = tmpPerson.xPos;
					dataBase[currentIndex].personObject.yPos = tmpPerson.yPos;
					dataBase[currentIndex].personObject.height = tmpPerson.height;
					dataBase[currentIndex].personObject.width = tmpPerson.width;
					dataBase[currentIndex].personObject.label = tmpPerson.label;
					memcpy(&dataBase[currentIndex].personObject.fv_array, &tmpPerson.fv_array, OUTPUT_SIZE*sizeof(float));
					dataBase[currentIndex].personObject.cameraID = tmpPerson.cameraID;
					dataBase[currentIndex].lru = 0;
					currentIndex++;
				}
				else{
					cout << "Database is full, please create a replacement policy." << endl;
				}
			}	
			else{
				pthread_mutex_unlock(&recvLocks[i]);	
			}	
		}  
	}

  for(int i = 0; i < NUM_THREADS; ++i) {
    pthread_join(listenThread, NULL);
  }


  return 0;
}*/


#include "stdio.h"
#include <iostream>
#include "stdlib.h"
#include <time.h>

using namespace std;

struct Info{
  //Mat hist;
  int cx;
  int cy;
  int h;
  int w;
};

struct uniqueID{
  uint16_t name;
  Info uInfo;
};

#define TABLESIZE   30

int main() {

srand (time(NULL));
int counter = 0;
string cameraID = "A";
struct uniqueID table[TABLESIZE];

//Populate table and send each entry to server
for (int i=0; i<TABLESIZE; i++) {
	table[i].name = i;
	table[i].uInfo.cx = rand() % 200 + 1;
	printf("Name: %d UniqueID: %d \n", table[i].name, table[i].uInfo.cx) ;
	//send to server
}

// Resend same table w/ indices reversed
for (int i = TABLESIZE-1; i >= 0; i--) {
	table[i].name = TABLESIZE-1-i;
	printf("Name: %d UniqueID: %d \n", table[i].name, table[i].uInfo.cx) ;
	//send to server
	//recieve new name
	//update new name
	//table[i].name = recieved_name;
}

return 0;
}

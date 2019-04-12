#ifndef __REID_GLOBALS_HPP__
#define __REID_GLOBALS_HPP__

#include "reid_structs.hpp"

float featureVectors[PINGPONG_SIZE][INPUT_BS*OUTPUT_SIZE+1];
float bboxes[BOX_PINGPONG_SIZE][INPUT_BS][4];
int stepPost = 0;
int stepBoxPost = 0;
int stepOutput = 0;
int stepBoxOutput = 0;
int openIndex = 0;
unsigned long long frame = 1000000;
array<int,3> colors[10];

object_history obj_table[NUM_OBJECTS];

#endif //__REID_GLOBALS_HPP__

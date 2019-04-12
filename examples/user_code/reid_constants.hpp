#ifndef __REID_CONSTANTS_HPP__
#define __REID_CONSTANTS_HPP__

#define NUM_OBJECTS 200
#define CAMERA_ID 1
#define EUC_THESH 5
#define IOU_THRESH 0.5
#define MULTI_MATCH_THRESH 2
#define SCORE_THRESH 0.6
#define EUC_WEIGHT 0.85 //0.9
#define IOU_WEIGHT 0.15 //0.1
#define PINGPONG_SIZE 1
#define BOX_PINGPONG_SIZE 1
#define KEY_SEND_THRESH 20
//e0.4 i0.6
//#define LIFE 30
#define LIFE 10

static const int INPUT_H = 256;
static const int INPUT_W = 128;
static const int INPUT_CH = 3;
static const int INPUT_BS = 16;
static const int OUTPUT_SIZE = 1280;
static const uint64_t MAX_ID_OFFSET = 1000000;

#endif //__REID_CONSTANTS_HPP__

#ifndef __REID_STRUCTS_HPP__
#define __REID_STRUCTS_HPP__

#include "reid_core_headers.hpp"
#include "reid_constants.hpp"

struct personType{
	int anomalyFlag;
	int currentCamera;
	int label;
	float fv_array[OUTPUT_SIZE];
	float xPos;
	float yPos;
	float height;
	float width;
};

struct reIDType{
	int oldID;
	int newID;
	//int newCameraID;
};

struct object_history{
	cv::Mat fv;
	int life;
	personType sendObject;
	int keyCount;
	int reIDFlag;
	int sentToServer;
};

struct ReIDBBox : public op::Datum
{
	std::vector<cv::Rect> personRectangle;
	std::vector<array<float,4>> personRectangleFloats;
	std::vector<int> keypointList;
	std::vector<cv::Mat> roi;
	std::vector<cv::Mat> mobileFv;
};

#endif //__REID_STRUCTS_HPP__

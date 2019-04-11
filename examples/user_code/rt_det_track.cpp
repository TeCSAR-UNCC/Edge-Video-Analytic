// Command-line user intraface
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

#include <opencv2/imgproc/imgproc.hpp> // cv::line, cv::circle
#include <opencv2/core/core.hpp>
#include <unistd.h>

/**********************************/
/******* Marhc 5, 2018**************/
#include <iterator>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <array>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "/usr/src/tensorrt/samples/common/common.h"



#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <pthread.h>

/**************************************/
/*           CLIENT INCLUDES          */
/**************************************/
#include "stdio.h"
#include "stdlib.h"
#include <time.h>

#include <unistd.h>
#include <sys/socket.h> 
#include <netinet/in.h> 
//#include <string.h> 
#include <arpa/inet.h>
#include <opencv2/opencv.hpp>

#include <queue>
/**************************************/

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

/*************************************/
/*          ADDING COMMS             */
/*************************************/
#define sPORT 76543
#define rPORT 76544
#define lIPADDR INADDR_ANY
#define NUM_THREADS 2
//#define OUTPUT_SIZE 1280
/**************************************/
static const int INPUT_H = 256;
static const int INPUT_W = 128;
static const int INPUT_CH = 3;
static const int INPUT_BS = 16;
static const int OUTPUT_SIZE = 1280;
using namespace nvinfer1;
IExecutionContext* context;
static Logger gLogger;
static int gUseDLACore{-1};
float featureVectors[PINGPONG_SIZE][INPUT_BS*OUTPUT_SIZE+1];
float bboxes[BOX_PINGPONG_SIZE][INPUT_BS][4];
int stepPost = 0;
int stepBoxPost = 0;
int stepOutput = 0;
int stepBoxOutput = 0;
int openIndex = 0;
unsigned long long frame = 1000000;
array<int,3> colors[10];

/*************************************/
/*          ADDING COMMS             */
/*************************************/
pthread_mutex_t locks[NUM_THREADS];
  
int sendLock = 0;
int recvLock = 1;

struct personType{
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

queue<personType> sendQ;
queue<reIDType> recvQ;

char * IPADDR;
/**************************************/

struct object_history{
	cv::Mat fv;
	int life;
	personType sendObject;
	int keyCount;
	int reIDFlag;
	int sentToServer;
};

object_history obj_table[NUM_OBJECTS];
/********************************/

struct UserDatum : public op::Datum
{
	std::vector<cv::Rect> personRectangle;
	std::vector<array<float,4>> personRectangleFloats;
	std::vector<int> keypointList;
	std::vector<cv::Mat> roi;
	std::vector<cv::Mat> mobileFv;
};

int currentLabel = 1000000;

/********************************************************/
/********************************************************/
/***************TensorRT Inference Functions*************/
/********************************************************/
/********************************************************/
const std::vector<std::string> directories{"/usr/src/tensorrt/data/samples/mnist/", "/usr/src/tensorrt/data/mnist/models/", "/home/tecsar/Edge-Video-Analytic/models/reid/"};
std::string locateFile(const std::string& input)
{
	return locateFile(input, directories);
}

void onnxToTRTModel(const std::string& modelFile, // name of the onnx model
	                unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
	                IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
	int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	auto parser = nvonnxparser::createParser(*network, gLogger);

	//Optional - uncomment below lines to view network layer information
	//config->setPrintLayerInfo(true);
	//parser->reportParsingInfo();

	if (!parser->parseFromFile(locateFile(modelFile, directories).c_str(), verbosity))
	{
	    string msg("failed to parse onnx file");
	    gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
	    exit(EXIT_FAILURE);
	}

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	/*************************/
	builder->setFp16Mode(true);
	builder->allowGPUFallback(true);
	/**************************/

	samplesCommon::enableDLA(builder, gUseDLACore);
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we can destroy the parser
	parser->destroy();

	// serialize the engine, then close everything down
	trtModelStream = engine->serialize();
	engine->destroy();
	network->destroy();
	builder->destroy();
}


// Handles collecting the pose bounding boxes
class WUserPostProcessing : public op::Worker<std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>>
{
public:
    WUserPostProcessing()
    {
        // User's constructor here
    }

    void initializationOnThread() {}

	/********************************************************/
	/********************************************************/
	/***************TensorRT Inference Functions*************/
	/********************************************************/
	/********************************************************/

	void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
	{
		const ICudaEngine& engine = context.getEngine();
		// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
		// of these, but in this case we know that there is exactly one input and one output.
		assert(engine.getNbBindings() == 2);
		void* buffers[2];

		// In order to bind the buffers, we need to know the names of the input and output tensors.
		// note that indices are guaranteed to be less than IEngine::getNbBindings()
		int inputIndex = -1;
		int outputIndex = -1;
		for (int b = 0; b < engine.getNbBindings(); ++b)
		{
		    if (engine.bindingIsInput(b))
		        inputIndex = b;
		    else
		        outputIndex = b;
		}

		// create GPU buffers and a stream
		CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_CH * INPUT_H * INPUT_W * sizeof(float)));
		CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

		cudaStream_t stream;
		CHECK(cudaStreamCreate(&stream));

		// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
		CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_CH * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
		context.enqueue(batchSize, buffers, stream, nullptr);
		CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
		cudaStreamSynchronize(stream);

		// release the stream and the buffers
		cudaStreamDestroy(stream);
		CHECK(cudaFree(buffers[inputIndex]));
		CHECK(cudaFree(buffers[outputIndex]));
	}

	/*******************************************************/
	/*******************************************************/
	/*******************************************************/

    void work(std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>& datumsPtr)
    {
    	std::vector<cv::Mat> inputCrops;
    	//cv::Mat image;
        try {
            if (datumsPtr != nullptr && !datumsPtr->empty()) {
                // Show in command line the resulting pose keypoints for body, face and hands
                timespec ts;
                clock_gettime(CLOCK_REALTIME, &ts); // Works on Linux
                // Accesing each element of the keypoints
                auto poseKeypoints = datumsPtr->at(0)->poseKeypoints;
                auto thresholdRectangle = 0.1f;
                auto thresholdKeypoints = 0.5f;
                auto minKeypoints = 15;

                for (auto person = 0 ; person < (std::min(poseKeypoints.getSize(0), INPUT_BS)); person++) {
                    /* Daniel's Edits: Resrict Re-ID to BB that have 12 keypoints or more*/
                    int keypointCount = op::getValidBBox(poseKeypoints,person, thresholdKeypoints, minKeypoints);
                    if(keypointCount >= minKeypoints ) {
                        auto rectBuffer = op::getKeypointsRectangle(poseKeypoints,person, thresholdRectangle); //Gets the rectangle information from the keypoints
                        array<float,4> arrayBuffer = {rectBuffer.x, rectBuffer.y, rectBuffer.width, rectBuffer.height};
                        datumsPtr->at(0)->personRectangleFloats.push_back(arrayBuffer);
                        datumsPtr->at(0)->keypointList.push_back(keypointCount);
                        cv::Rect rec(rectBuffer.x,rectBuffer.y,rectBuffer.width,rectBuffer.height); //Converts from openpose rectangle to opencv rectangle

                        bboxes[stepBoxPost][person][0] = rectBuffer.x;
                        bboxes[stepBoxPost][person][1] = rectBuffer.y;
                        bboxes[stepBoxPost][person][2] = rectBuffer.width;
                        bboxes[stepBoxPost][person][3] = rectBuffer.height;

                        op::keepRoiInside(rec,datumsPtr->at(0)->cvInputData.cols,datumsPtr->at(0)->cvInputData.rows);
                        if((rec.width > 0) && (rec.height > 0) && (rec.x < datumsPtr->at(0)->cvInputData.cols) && (rec.y < datumsPtr->at(0)->cvInputData.rows)) {
                            if(rec.x+rec.width > datumsPtr->at(0)->cvInputData.cols)
                                rec.width = datumsPtr->at(0)->cvInputData.cols - rec.x - 1;
                            if(rec.y+rec.height > datumsPtr->at(0)->cvInputData.rows)
                                rec.height = datumsPtr->at(0)->cvInputData.rows - rec.y - 1;
                            int img_width = datumsPtr->at(0)->cvInputData.cols;
                            int img_height = datumsPtr->at(0)->cvInputData.rows;
                            rec.x = max(rec.x,0);
                            rec.y = max(rec.y,0);
                            rec.width = min(rec.width, img_width-rec.x-1);
                            rec.height = min(rec.height, img_height-rec.y-1);
                            datumsPtr->at(0)->personRectangle.push_back(rec); //Inserts the rectangle into the datums vector: personRectangle.
                            cv::Mat img = datumsPtr->at(0)->cvInputData(rec);
                            cv::Size s = img.size();
                            if(s.width > 0 && s.height > 0) {
                                cv::resize(img, img, cv::Size(128, 256));
                                inputCrops.push_back(img);
                            }
                        }
                    }
                }

                float data[INPUT_BS][INPUT_CH][INPUT_H][INPUT_W] ={0};

                std::vector<cv::Mat> p;
                cv::Mat fill(cv::Size(128, 256), CV_32FC1, cv::Scalar(0)); 
                p.push_back(fill);
                p.push_back(fill);
                p.push_back(fill);

                float mean[3] = {0.486,0.459,0.408};
                float stdDev[3] = {0.229,0.224,0.225};
                int vectorSize = inputCrops.size();
                for(int b = 0; b < INPUT_BS; b++) {
                    if(b < vectorSize) {
                        cv::split(inputCrops[b], p);
                    }
                    for(int c = 0; c < INPUT_CH; ++c) {
                        for(int h = 0; h < INPUT_H; ++h) {
                            for(int w = 0; w < INPUT_W; ++w) {
                                data[b][c][h][w] = ((((float)p[2-c].at<uchar>(h, w))/255)-mean[c])/stdDev[c];
                            }
                        }
                    }
                }

                // run inference
                float* dataP = (float*)data;
                float output[OUTPUT_SIZE*INPUT_BS];
                doInference(*context, dataP, output, INPUT_BS);

                std::copy(std::begin(output), std::end(output), std::begin(featureVectors[stepPost]));

                featureVectors[stepPost][INPUT_BS*OUTPUT_SIZE] = (float)vectorSize;

                for (auto i = 0 ; (unsigned)i < datumsPtr->at(0)->personRectangle.size(); i++) {
                    cv::Mat detection(1,OUTPUT_SIZE,CV_32F);
                    void* ptr = &featureVectors[stepOutput][i*OUTPUT_SIZE];
                    std::memcpy(detection.data, ptr, OUTPUT_SIZE*sizeof(float));
                    datumsPtr->at(0)->mobileFv.push_back(detection);
                }

                if(stepPost < PINGPONG_SIZE-1) {
                    stepPost++;
                } else {
                    stepPost = 0;
                }

                if(stepBoxPost < BOX_PINGPONG_SIZE-1) {
                    stepBoxPost++;
                } else {
                    stepBoxPost = 0;
                }
            }
        } catch (const std::exception& e) {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }

    }
};

// Handles the output image
class WUserOutput : public op::WorkerConsumer<std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>>
{
public:
    void initializationOnThread() {}

    float intersectionOverUnion(float box1[4], float box2[4])
    {
        //std::cout << "box1\n";
	    float minx1 = box1[0];
	    float maxx1 = box1[0] + box1[2];
	    float miny1 = box1[1];
	    float maxy1 = box1[1] + box1[3];

	    float minx2 = box2[0];
	    float maxx2 = box2[0] + box2[2];
	    float miny2 = box2[1];
	    float maxy2 = box2[1] + box2[3];

	    if (minx1 > maxx2 || maxx1 < minx2 || miny1 > maxy2 || maxy1 < miny2) {
		    return 0.0f;
	    } else {
		    float dx = std::min(maxx2, maxx1) - std::max(minx2, minx1);
		    float dy = std::min(maxy2, maxy1) - std::max(miny2, miny1);
		    float area1 = (box1[2])*(box1[3]);
		    float area2 = (box2[2])*(box2[3]);
		    float inter = dx*dy; // Intersection
		    float uni = area1 + area2 - inter; // Union
		    float IoU = inter / uni;
		    return IoU;
	    }
    }

    /*****************************************/
    void setLabel(cv::Mat& im, const std::string label, const cv::Point & point, int cIndex)
    {
        int fontface = cv::FONT_HERSHEY_SIMPLEX;
        double scale = 0.7;
        int thickness = 2;
        int baseline = 0;

        cv::Size text = cv::getTextSize(label, fontface, scale, thickness, &baseline);
        cv::rectangle(im, point + cv::Point(0, baseline), point + cv::Point(text.width, -text.height), CV_RGB(colors[cIndex][0],colors[cIndex][1],colors[cIndex][2]), CV_FILLED);
        cv::putText(im, label, point, fontface, scale, CV_RGB(0,0,0), thickness, 8);
    }
    /*************************************************/
    /********************* Algorithm Functions ************************************/
    int findBest(vector<array<float,2>> inputVector, vector<int> vectorDone) {
	    int best = -1;
	    float tempScore = 1000;
	    int cont = 0;
	    for (int i = 0; i < (int)inputVector.size(); i++) {
		    cont = 0;
		    if (!vectorDone.empty()) {
			    for (int j = 0; j < (int)vectorDone.size(); j++){
				    if (vectorDone.at(j) == (int)(inputVector.at(i))[0]) {
					    cont = 1;
				    }
			    }
		    }
		    if (!cont) {
			    if(tempScore > inputVector.at(i)[1]){
				    best = (int)inputVector.at(i)[0];
				    tempScore = inputVector.at(i)[1];
			    }
		    }
	    }
	    return best;
    }
    /******************************************************************************/

    void workConsumer(const std::shared_ptr<std::vector<std::shared_ptr<UserDatum>>>& datumsPtr) {
        //take output array
        //iterate through (till number of detections)
        //convert each portion to to a mat
        //do l2 norm against high IoU
        //with l2 norm and IoU scores:
        //make new label or replace in table

        try {
            if (datumsPtr != nullptr) {
                if(frame != datumsPtr->at(0)->frameNumber) {
                    for(int j = 0; j < NUM_OBJECTS; ++j) {
                        if (obj_table[j].life > 0){
                            obj_table[j].life--;
                            if((obj_table[j].life == 0)  && (obj_table[j].sentToServer == 1)) {
                                obj_table[j].sendObject.currentCamera = -1;
                                pthread_mutex_lock(&locks[sendLock]);
                                sendQ.push(obj_table[j].sendObject);
                                pthread_mutex_unlock(&locks[sendLock]);
                            }
                        }
                    }
                }
                frame = datumsPtr->at(0)->frameNumber;
            }
            if (datumsPtr != nullptr && !datumsPtr->empty()) {
                reIDType tmpReIDType;
                for(int i = 0; i < 3; ++i) {
                    if(!recvQ.empty()) { 
                        tmpReIDType = recvQ.front(); 
                        std::cout << "Received tmpReIDType.old = " << tmpReIDType.oldID << std::endl;
                        recvQ.pop();
                        for(int j = 0; j < NUM_OBJECTS; ++j) {
                            if(obj_table[j].sendObject.label == tmpReIDType.oldID) {
                                //replace and break
                                obj_table[j].sendObject.label = tmpReIDType.newID;
                                obj_table[j].reIDFlag = 1;
                                j = NUM_OBJECTS;
                            }
                        }
                    }
                }

                int id_labels[16][2];
                /* Algorithm implementation: Setup */
                vector<vector<array<float,2>>> detectionVector;
                vector<array<float,2>> tableVector[NUM_OBJECTS];
                std::vector<std::vector<int>> flagMulti;
                vector<int> tableDone;
                vector<int> detDone;
                vector<int> newDetect;
                vector<int> openTableIdx;
                vector<array<int,2>> bestMatches;
                float opBBox[4];
                for (auto i = 0 ; (unsigned)i < datumsPtr->at(0)->personRectangle.size(); i++) {
                    opBBox[0] = datumsPtr->at(0)->personRectangleFloats.at(i)[0];
                    opBBox[1] = datumsPtr->at(0)->personRectangleFloats.at(i)[1];
                    opBBox[2] = datumsPtr->at(0)->personRectangleFloats.at(i)[2];
                    opBBox[3] = datumsPtr->at(0)->personRectangleFloats.at(i)[3];
                    cv::Mat detection(1,OUTPUT_SIZE,CV_32F);
                    void* ptr = &featureVectors[stepOutput][i*OUTPUT_SIZE];
                    std::memcpy(detection.data, ptr, OUTPUT_SIZE*sizeof(float));
                    vector<array<float,2>> currentDetection;
                    vector<int> currentMulti;
                    for(int j = 0; j < NUM_OBJECTS; ++j) {
                        if (obj_table[j].life > 0){
                            double dopEucDist = cv::norm(obj_table[j].fv,datumsPtr->at(0)->mobileFv.at(i), cv::NORM_L2);
                            float opEucDist = (float)dopEucDist;
                            float box1[4];
                            box1[0] = obj_table[j].sendObject.xPos;
                            box1[1] = obj_table[j].sendObject.yPos;
                            box1[2] = obj_table[j].sendObject.width;
                            box1[3] = obj_table[j].sendObject.height;
                            float opIoU = intersectionOverUnion(box1,opBBox);

                            float opScore = (opEucDist/10)*EUC_WEIGHT + (1-opIoU)*IOU_WEIGHT;

                            if (opScore < SCORE_THRESH) {
                                /* Algorithm implementation: Implementation */
                                array<float,2> tempHolder = {(float)j,opScore};
                                currentDetection.push_back(tempHolder);
                                tempHolder = {(float)i,opScore};
                                tableVector[j].push_back(tempHolder);
                                currentMulti.push_back(j);
                            }
                        }
                        else if(i==0) {
                            openTableIdx.push_back(j);
                        }
                    }
                    detectionVector.push_back(currentDetection);
                    flagMulti.push_back(currentMulti);
                }
                int prevSize = detDone.size() - 1;
                while( (detDone.size() < detectionVector.size()) && ((int)detDone.size() != prevSize)) {
                    prevSize = detDone.size();
                    for (int i = 0; i < (int)detectionVector.size(); i++) {
                        int index = findBest(detectionVector.at(i),tableDone);
                        if (index >= 0 && index < NUM_OBJECTS) {
                            if (i == findBest(tableVector[index],detDone)) {
                                detDone.push_back(i);
                                tableDone.push_back(index);
                            }
                        }
                    }
                }
                if (detDone.size() < detectionVector.size()) {
                    //if table entry does not exist in detDone, it is new
                    for(int i = 0; i < (int)detectionVector.size(); ++i) {
                        std::vector<int>::iterator it;
                        it = find(detDone.begin(), detDone.end(),i);
                        if (it == detDone.end()) {
                            newDetect.push_back(i);
                        }
                    }
                }

                for (int i = 0; i < (int)detDone.size(); ++i) {
                    int eucIndex = tableDone.at(i);
                    if (((datumsPtr->at(0)->keypointList.at(detDone.at(i)) > obj_table[eucIndex].keyCount) || ((datumsPtr->at(0)->keypointList.at(detDone.at(i)) >= KEY_SEND_THRESH) && 
                        (obj_table[eucIndex].sentToServer == 0) )) && (flagMulti.at(detDone.at(i)).size() < MULTI_MATCH_THRESH)) {
                        int updateFeatureFlag = 1;
                        float checkBox[4];
                        checkBox[0] = obj_table[detDone.at(i)].sendObject.xPos;
                        checkBox[1] = obj_table[detDone.at(i)].sendObject.yPos;
                        checkBox[2] = obj_table[detDone.at(i)].sendObject.width;
                        checkBox[3] = obj_table[detDone.at(i)].sendObject.height;
                        for(int j = 0; j < NUM_OBJECTS; ++j) {
                            if(obj_table[j].life > 0) {
                                if (detDone.at(i) == j) continue;
                                float box1[4];
                                box1[0] = obj_table[j].sendObject.xPos;
                                box1[1] = obj_table[j].sendObject.yPos;
                                box1[2] = obj_table[j].sendObject.width;
                                box1[3] = obj_table[j].sendObject.height;
                                float tmpIoU = intersectionOverUnion(box1, checkBox);
                                if (tmpIoU > 0.3f) {
                                    updateFeatureFlag = 0;
                                }
                            }
                        }
                        if (updateFeatureFlag ==1) {
                            std::cout << "Updated: Det|Tab\t" << detDone.at(i) << "|" << eucIndex << std::endl;
                            void* ptr = &featureVectors[stepOutput][detDone.at(i)*OUTPUT_SIZE];
                            std::memcpy(obj_table[eucIndex].fv.data, ptr, OUTPUT_SIZE*sizeof(float));
                            std::memcpy(&obj_table[eucIndex].sendObject.fv_array, &featureVectors[stepOutput][detDone.at(i)*OUTPUT_SIZE], OUTPUT_SIZE*sizeof(float));
                            obj_table[eucIndex].keyCount = datumsPtr->at(0)->keypointList.at(detDone.at(i));
                            if ((datumsPtr->at(0)->keypointList.at(detDone.at(i)) > KEY_SEND_THRESH) || ((obj_table[eucIndex].sentToServer == 0) &&
                                (datumsPtr->at(0)->keypointList.at(detDone.at(i)) >= KEY_SEND_THRESH))) {
                                //put in send Q
                                std::cout << "Sent: Det|Tab\t" << detDone.at(i) << "|" << eucIndex << std::endl;
                                obj_table[eucIndex].sendObject.currentCamera = CAMERA_ID;
                                obj_table[eucIndex].sentToServer = 1;
                                pthread_mutex_lock(&locks[sendLock]);
                                sendQ.push(obj_table[eucIndex].sendObject);
                                pthread_mutex_unlock(&locks[sendLock]);
                            }
                        }
                    }
                    obj_table[eucIndex].sendObject.xPos = bboxes[stepBoxOutput][detDone.at(i)][0];
                    obj_table[eucIndex].sendObject.yPos = bboxes[stepBoxOutput][detDone.at(i)][1];
                    obj_table[eucIndex].sendObject.width = bboxes[stepBoxOutput][detDone.at(i)][2];
                    obj_table[eucIndex].sendObject.height = bboxes[stepBoxOutput][detDone.at(i)][3];
                    obj_table[eucIndex].life = LIFE;
                    id_labels[detDone.at(i)][0] = obj_table[eucIndex].sendObject.label;
                    id_labels[detDone.at(i)][1] = obj_table[eucIndex].reIDFlag;
                }
                for(int i = 0; i < (int)newDetect.size(); ++i) {
                    if(i < (int)openTableIdx.size()) {
                        int eucIndex = openTableIdx.at(i);
                        void* ptr = &featureVectors[stepOutput][newDetect.at(i)*OUTPUT_SIZE];
                        std::memcpy(obj_table[eucIndex].fv.data, ptr, OUTPUT_SIZE*sizeof(float));
                        std::memcpy(&obj_table[eucIndex].sendObject.fv_array, &featureVectors[stepOutput][newDetect.at(i)*OUTPUT_SIZE], OUTPUT_SIZE*sizeof(float));
                        obj_table[eucIndex].keyCount = datumsPtr->at(0)->keypointList.at(newDetect.at(i));
                        obj_table[eucIndex].sendObject.xPos = bboxes[stepBoxOutput][newDetect.at(i)][0];
                        obj_table[eucIndex].sendObject.yPos = bboxes[stepBoxOutput][newDetect.at(i)][1];
                        obj_table[eucIndex].sendObject.width = bboxes[stepBoxOutput][newDetect.at(i)][2];
                        obj_table[eucIndex].sendObject.height = bboxes[stepBoxOutput][newDetect.at(i)][3];
                        obj_table[eucIndex].sendObject.currentCamera = CAMERA_ID;
                        obj_table[eucIndex].sentToServer = 0;
                        obj_table[eucIndex].life = LIFE;
                        obj_table[eucIndex].reIDFlag = 0;
                        if(currentLabel == 3000000){
                            currentLabel = 2000000;
                        }
                        obj_table[eucIndex].sendObject.label = currentLabel;
                        id_labels[newDetect.at(i)][0] = currentLabel;
                        id_labels[newDetect.at(i)][1] = 0;
                        currentLabel++;
                    } else {
                        id_labels[newDetect.at(i)][0] = -1;
                        id_labels[newDetect.at(i)][1] = 0;
                        break;
                    }
                }


                if(stepOutput < PINGPONG_SIZE-1) {
                    stepOutput++;
                } else {
                    stepOutput = 0;
                }

                if(stepBoxOutput < BOX_PINGPONG_SIZE-1) {
                    stepBoxOutput++;
                } else {
                    stepBoxOutput = 0;
                }

                for (auto person = 0 ; (unsigned)person < datumsPtr->at(0)->personRectangle.size(); person++) {
                    //A complicated if statement to make sure that a random detection out of bounds is not made
                    if ((datumsPtr->at(0)->personRectangle[person].width > 0) && (datumsPtr->at(0)->personRectangle[person].height > 0) &&
                        (datumsPtr->at(0)->personRectangle[person].x < datumsPtr->at(0)->cvOutputData.cols) &&
                        (datumsPtr->at(0)->personRectangle[person].y < datumsPtr->at(0)->cvOutputData.rows)) {
                        std::string tmpStringLabel = std::to_string(id_labels[person][0]);
                        std::string camera(1, tmpStringLabel.at(0));
                        int labelInt = std::stoi(tmpStringLabel.substr(1));
                        std::string label = std::to_string(labelInt);
                        std::string finalLabel = "P" + label + "-" + camera;

                        int colorIndex = 9;
                        if(id_labels[person][1] == 1) {
                            colorIndex = labelInt%9;
                        }

                        cv::rectangle(datumsPtr->at(0)->cvOutputData, datumsPtr->at(0)->personRectangle[person],
                                      cv::Scalar(colors[colorIndex][2],colors[colorIndex][1],colors[colorIndex][0]),2); //Draws the bounding box around the peson of interest

                        setLabel(datumsPtr->at(0)->cvOutputData, finalLabel, cv::Point(datumsPtr->at(0)->personRectangle[person].x,datumsPtr->at(0)->personRectangle[person].y), colorIndex);
                    }
                }
            }
        } catch (const std::exception& e) {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

int generateBoundingBoxes()
{
    try
    {
        op::log("Starting OpenPose demo...", op::Priority::High);
        const auto timerBegin = std::chrono::high_resolution_clock::now();

        // logging_level
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                  __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // cameraSize
        const auto cameraSize = op::flagsToPoint(FLAGS_camera_resolution, "-1x-1");
        // outputSize
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        // producerType
        op::ProducerType producerType;
        std::string producerString;
        std::tie(producerType, producerString) = op::flagsToProducer(
            FLAGS_image_dir, FLAGS_video, FLAGS_ip_camera, FLAGS_camera, FLAGS_flir_camera, FLAGS_flir_camera_index);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::log("Flag `write_keypoint` is deprecated and will eventually be removed."
                    " Please, use `write_json` instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
                                                      FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1 || FLAGS_flir_camera);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        const auto poseMode = op::flagsToPoseMode(FLAGS_body);

        // Configuring OpenPose
        op::log("Configuring OpenPose...", op::Priority::High);
		op::WrapperT<UserDatum> opWrapperT; //{op::ThreadManagerMode::Asynchronous};

        // Initializing the user custom classes
        // Processing
        auto wUserPostProcessing = std::make_shared<WUserPostProcessing>();
        // Add custom processing
        const auto workerProcessingOnNewThread = true;
        opWrapperT.setWorker(op::WorkerType::PostProcessing, wUserPostProcessing, workerProcessingOnNewThread);

        /********************************************************/
	    // Initializing the user custom classes
        // Processing
        auto wUserOutput = std::make_shared<WUserOutput>();
        // Add custom processing
        opWrapperT.setWorker(op::WorkerType::Output, wUserOutput, workerProcessingOnNewThread);

        /********************************************************/
        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, outputSize, keypointScaleMode, FLAGS_num_gpu, FLAGS_num_gpu_start,
            FLAGS_scale_number, (float)FLAGS_scale_gap, op::flagsToRenderMode(FLAGS_render_pose, multipleView),
            poseModel, !FLAGS_disable_blending, (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap,
            FLAGS_part_to_show, FLAGS_model_folder, heatMapTypes, heatMapScaleMode, FLAGS_part_candidates,
            (float)FLAGS_render_threshold, FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max,
            FLAGS_prototxt_path, FLAGS_caffemodel_path, (float)FLAGS_upsampling_ratio, enableGoogleLogging};
        opWrapperT.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{};
        opWrapperT.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{};
        opWrapperT.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{};
        opWrapperT.configure(wrapperStructExtra);
        // Producer (use default to disable any input)
        const op::WrapperStructInput wrapperStructInput{
            producerType, producerString, FLAGS_frame_first, FLAGS_frame_step, FLAGS_frame_last,
            FLAGS_process_real_time, FLAGS_frame_flip, FLAGS_frame_rotate, FLAGS_frames_repeat,
            cameraSize, FLAGS_camera_parameter_path, FLAGS_frame_undistort, FLAGS_3d_views};
        opWrapperT.configure(wrapperStructInput);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, FLAGS_write_keypoint, op::stringToDataFormat(FLAGS_write_keypoint_format),
            FLAGS_write_json, FLAGS_write_coco_json, FLAGS_write_coco_json_variants, FLAGS_write_coco_json_variant,
            FLAGS_write_images, FLAGS_write_images_format, FLAGS_write_video, FLAGS_write_video_fps,
            FLAGS_write_video_with_audio, FLAGS_write_heatmaps, FLAGS_write_heatmaps_format, FLAGS_write_video_3d,
            FLAGS_write_video_adam, FLAGS_write_bvh, FLAGS_udp_host, FLAGS_udp_port};
        opWrapperT.configure(wrapperStructOutput);
        // GUI (comment or use default argument to disable any visual output)
        const op::WrapperStructGui wrapperStructGui{
            op::flagsToDisplayMode(FLAGS_display, FLAGS_3d), !FLAGS_no_gui_verbose, FLAGS_fullscreen};
        opWrapperT.configure(wrapperStructGui);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapperT.disableMultiThreading();
        // Start, run, and stop processing - exec() blocks this thread until OpenPose wrapper has finished
        op::log("Starting thread(s)...", op::Priority::High);
        opWrapperT.exec();

        // Measuring total time
        const auto now = std::chrono::high_resolution_clock::now();
        const auto totalTimeSec = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(now-timerBegin).count()
                                * 1e-9;
        const auto message = "OpenPose demo successfully finished. Total time: "
                           + std::to_string(totalTimeSec) + " seconds.";
        op::log(message, op::Priority::High);

        // Return successful message
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}



/*************************************/
/*          ADDING COMMS             */
/*************************************/

void *sendData(void*) {
    /* Initializing network parameters */
    int sock;
    //char* IPADDR = "192.168.0.26";
    struct sockaddr_in servaddr;

    sock = socket(AF_INET,SOCK_STREAM,0);
    bzero(&servaddr,sizeof servaddr);

    servaddr.sin_family=AF_INET;
    servaddr.sin_port=htons(sPORT);

    inet_pton(AF_INET, IPADDR, &(servaddr.sin_addr));

    connect(sock,(struct sockaddr *)&servaddr,sizeof(servaddr)); 

    personType data;
    std::cout << "Entering Send function." << std::endl;

    while(1) {
        pthread_mutex_lock(&locks[sendLock]);
        if(!sendQ.empty()) {
            std::cout << "Inside send Queue\n";
            data = sendQ.front();
            sendQ.pop();
            pthread_mutex_unlock(&locks[sendLock]);
            cout << "data.label = " << data.label << endl;

            int status = send(sock,reinterpret_cast<const char*> (&data), sizeof(personType), 0);
            cout << "Sent Data: " << status << endl;
        }
        else{
            pthread_mutex_unlock(&locks[sendLock]);
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

        reIDType data;
        int status = 1;

        while(status > 0) {
            status = recv(comm_fd, &data, sizeof(reIDType), 0);

            pthread_mutex_lock(&locks[recvLock]);
            recvQ.push(data);
	            cout << "data.newID = " << data.newID << endl;
            pthread_mutex_unlock(&locks[recvLock]);

            cout << " status: " << status << endl;
        }
        close(comm_fd);
        close(listen_fd);
    }
}
/*************************************/


int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    gUseDLACore = op::flagsToDLA(FLAGS_use_dla);
    // create a TensorRT model from the onnx model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};
    onnxToTRTModel("mobileNet_single.onnx", (INPUT_BS), trtModelStream);
    assert(trtModelStream != nullptr);


    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gUseDLACore >= 0)
    {
        runtime->setDLACore(gUseDLACore);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    context = engine->createExecutionContext();
    assert(context != nullptr);

	colors[0] = {255,0,0};
	colors[1] = {0,255,0};
	colors[2] = {0,0,255};
	colors[3] = {255,255,0};
	colors[4] = {255,0,255};
	colors[5] = {0,255,255};
	colors[6] = {177,177,0};
	colors[7] = {177,0,177};
	colors[8] = {0,177,177};
	colors[9] = {177,177,177};

    /*************************************/
    /*          ADDING COMMS             */
    /*************************************/
    IPADDR = (char *)(FLAGS_server_ip.c_str());
    pthread_t threads[NUM_THREADS];

    for(int i = 0; i < NUM_THREADS; ++i) {
        pthread_mutex_init(&locks[i], NULL);
    }

    for(int i = 0; i < NUM_THREADS; ++i) {
        pthread_create(&threads[i], NULL, sendData, NULL);
        ++i;
        pthread_create(&threads[i], NULL, recvData, NULL);
    }
    /*************************************/

	for(int i = 0; i < NUM_OBJECTS; ++i) {
		obj_table[i].sendObject.currentCamera = CAMERA_ID;
		obj_table[i].sendObject.label = 0;
		for(int j = 0; j < OUTPUT_SIZE; ++j) {
			obj_table[i].sendObject.fv_array[j] = 0;
		}
		obj_table[i].fv = cv::Mat::zeros(1, OUTPUT_SIZE, CV_32F);
		obj_table[i].sendObject.xPos = 0;
		obj_table[i].sendObject.yPos = 0;
		obj_table[i].sendObject.height = 0;
		obj_table[i].sendObject.width = 0;
		obj_table[i].life = 0;
		obj_table[i].sentToServer = 0;
	}
    generateBoundingBoxes();
	context->destroy();
	engine->destroy();
	runtime->destroy();

	/*************************************/
	/*          ADDING COMMS             */
	/*************************************/
	for(int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }
    /*************************************/

	return 0;
}

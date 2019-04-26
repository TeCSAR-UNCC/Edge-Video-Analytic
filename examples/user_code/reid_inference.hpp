#ifndef __REID_INFERENCE_HPP__
#define __REID_INFERENCE_HPP__

#include "reid_core_headers.hpp"
#include "trt_helper.hpp"

// Handles collecting the pose bounding boxes
class ReIDInference : public op::Worker<std::shared_ptr<std::vector<std::shared_ptr<ReIDBBox>>>> {
  public:
    ReIDInference()
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

    void work(std::shared_ptr<std::vector<std::shared_ptr<ReIDBBox>>>& datumsPtr)
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
					//print keypoints
					//std::cout << poseKeypoints.toString();
                    if(keypointCount >= minKeypoints ) {
                        auto rectBuffer = op::getKeypointsRectangle(poseKeypoints,person, thresholdRectangle); //Gets the rectangle information from the keypoints
                        array<float,4> arrayBuffer = {rectBuffer.x, rectBuffer.y, rectBuffer.width, rectBuffer.height};
                        datumsPtr->at(0)->personRectangleFloats.push_back(arrayBuffer);
                        datumsPtr->at(0)->keypointNumPerPerson.push_back(keypointCount);
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
							datumsPtr->at(0)->keypointIndex.push_back(person);
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

#endif //__REID_INFERENCE_HPP__

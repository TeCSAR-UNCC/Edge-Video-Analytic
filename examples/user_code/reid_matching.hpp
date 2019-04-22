#ifndef __REID_MATCHING_HPP__
#define __REID_MATCHING_HPP__

#include "reid_core_headers.hpp"
#include "reid_globals.hpp"
#include "reid_constants.hpp"
#include "reid_comms.hpp"

// Handles the output image
class ReIDMatching : public op::WorkerConsumer<std::shared_ptr<std::vector<std::shared_ptr<ReIDBBox>>>>
{
private:
    uint64_t BASE_LABEL;
    uint64_t currentLabel;
public:
    ReIDMatching() {
        BASE_LABEL = 1000000;
        currentLabel = BASE_LABEL;
    }

    ReIDMatching(uint64_t base) {
        BASE_LABEL = base;
        currentLabel = BASE_LABEL;
    }

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

    void workConsumer(const std::shared_ptr<std::vector<std::shared_ptr<ReIDBBox>>>& datumsPtr) {
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
								auto poseKeypoints = datumsPtr->at(0)->poseKeypoints;
								auto thresholdKeypoints = 0.5f;
								obj_table[eucIndex].sendObject.anomalyFlag = anomalySend(poseKeypoints,detDone.at(i), thresholdKeypoints);
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
                        if(currentLabel == BASE_LABEL+MAX_ID_OFFSET){
                            currentLabel = BASE_LABEL;
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
		                    cv::rectangle(datumsPtr->at(0)->cvOutputData, datumsPtr->at(0)->personRectangle[person],
		                                  cv::Scalar(colors[colorIndex][2],colors[colorIndex][1],colors[colorIndex][0]),2); //Draws the bounding box around the peson of interest

		                    setLabel(datumsPtr->at(0)->cvOutputData, finalLabel, cv::Point(datumsPtr->at(0)->personRectangle[person].x,datumsPtr->at(0)->personRectangle[person].y), colorIndex);
                        }
                    }
                }
            }
        } catch (const std::exception& e) {
            this->stop();
            op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
};

#endif //__REID_MATCHING_HPP__

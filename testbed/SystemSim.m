clear
clc

fullValidation = 1;
cameras = [5];
camera_ranges = zeros(length(cameras),2);

load('data/ground_truth/mtmc_gt.mat');
if (fullValidation == 1)
    for cam = cameras
        cam_gt = mtmc_gt(mtmc_gt(:,1)==cam,:);
        camera_ranges(cameras==cam,:) = [min(cam_gt(:,3)), max(cam_gt(:,3))];
    end
    test_range = [min(camera_ranges(:,1)), max(camera_ranges(:,2))];
    clear cam_gt;
else
    for cam = cameras
        camera_ranges(cameras==cam,:) = [100000, 110000];
    end
end
clear cam;

data_path = 'data/DukeMTMC/detections';

for i = 1:length(cameras)
    edge_nodes(i) = EdgeNode(camera_params(cameras(i),data_path, ...
                             camera_ranges(i,1), ...
                             camera_ranges(i,2), ...
                             60,5,200,10,0.5,15));
end
%%

for i = 1:length(cameras)
    edge_nodes(i) = edge_nodes(i).resetNode(); % Reset Edge Nodes
end

edge_server_ops(edge_nodes,1); % Reset Edge Server

for i = test_range(1):test_range(2)
    for n = 1:length(cameras)
        if (edge_nodes(n).ready(i)==1)
            edge_nodes(n) = edge_nodes(n).process_step();
        end
    end
    
    edge_nodes = edge_server_ops(edge_nodes,0);
end


function r = camera_params(id, dataPath, startFrame, endFrame, srcFrameRt, outFrameRt, tabSize, tabLife, kpcth, kpcnt)
% CAMERA_PARAMS Helper function for generating camera_params struct
%   Inputs:
%       id - Camera ID #, used for finding correct input mats as well as
%       label generation
%       dataPath - Directory path to detection mats
%       startFrame - Starting frame # for eval on the camera
%       endFrame - Ending frame # for eval on the camera
%       srcFrameRate - Frame rate at which the original data was recorded
%       outFrameRate - Frame rate at which the node will operate. Use to
%           determine which frames will be processed
%       tabSize - Size of the identity table maintained on the node
%       kpcth - Keypoint confidence threshold for determining valid
%           keypoints in OpenPose
%   Output:
%       r - camera_params struct

    if nargin > 1
        frame_interval = ceil(srcFrameRt/outFrameRt);
        frames = (startFrame:frame_interval:endFrame);
        dets = load(strcat(dataPath,'/tecsar/camera',int2str(id),'.mat'));
        test_rows = sum((dets.detections(:,1) == frames),2);
        cam_dets = dets.detections(test_rows==1,:);
        clear dets;
        feats = load(strcat(dataPath,'/features/camera',int2str(id),'.mat'));
        cam_feats = feats.reid_features(test_rows==1,:);
        clear test_rows feats;
    else
        id = 0;
        cam_dets = zeros(1,76);
        cam_feats = zeros(1,1281);
        srcFrameRt = 0;
        outFrameRt = 1;
        startFrame = 0;
        endFrame = 0;
        tabSize = 1;
        kpcth = 0;
        kpcnt = 0;
        tabLife = 1;
    end
    
    r = struct('id',id,'dets',cam_dets,'feats',cam_feats,'sourceFrameRate',srcFrameRt, ...
               'outFrameRate',outFrameRt,'startFrame',startFrame, ...
               'endFrame',endFrame,'tab_size',tabSize,'kpcth',kpcth, ...
               'kpcnt',kpcnt,'tabLife',tabLife);
end

function r = object_history(personObject, lru)
    r = struct('personObject',personObject,'lru',lru);
end

function nodes = edge_server_ops(nodes, rst)
    if nargin == 1
        rst = 0;
    elseif nargin == 0
        error('Expecting at least 1 argument');
    end

    persistent ReID_table;
    persistent currentIndex;
    
    DBSIZE = 1000;
    L2_THR = 4.25;
    
    if (isempty(ReID_table) || (rst==1))
        pt = personType();
        oh = object_history(pt, 1000);
        ReID_table = repmat(oh,DBSIZE,1);
        currentIndex = 1;
        return;
    end
    
    for i = 1:DBSIZE
        ReID_table(i).lru = ReID_table(i).lru + 1;
    end
    
    sendQ = [];
    
    for cam = 1:size(nodes,1)
        queue = nodes(cam).getSendQ();
    
        if (~isempty(queue))
            for q = 1:length(queue)
                tmpPerson = queue(q);
                matchIdx = -1;
                minL2 = Inf;
                updateFlag = 0;
                max = -1;
                
                for i = 1:DBSIZE
                    ReID_person = ReID_table(i).personObject;
                    if (tmpPerson.currentCamera > -1)
                        match = norm(tmpPerson.fv_array - ReID_person.fv_array);
                        if ((ReID_person.label == tmpPerson.label) && ...
                            (ReID_person.currentCamera == tmpPerson.currentCamera))
                            updateFlag = 1;
                            matchidx = i;
                            break;
                        elseif ((match < minL2) && ReID_person.currentCamera == -1)
                            minL2 = match;
                            matchIdx = i;
                        end
                    elseif (ReID_person.label == tmpPerson.label)
                        ReID_table(i).personObject.currentCamera = -1;
                        break;
                    end
                end

                if ((tmpPerson.currentCamera > -1) && (updateFlag == 0))
                    if ((matchIdx > 0) && (minL2 < L2_THR))
                        ReID_table(matchIdx).personObject.xPos = tmpPerson.xPos;
                        ReID_table(matchIdx).personObject.yPos = tmpPerson.yPos;
                        ReID_table(matchIdx).personObject.width = tmpPerson.width;
                        ReID_table(matchIdx).personObject.height = tmpPerson.height;
                        ReID_table(matchIdx).personObject.currentCamera = tmpPerson.currentCamera;
                        ReID_table(matchIdx).personObject.fv_array = tmpPerson.fv_array;
                        ReID_table(matchIdx).lru = 0;
                        sendQ = [sendQ; reIDType(tmpPerson.label,ReID_table(matchIdx).personObject.label)];
                    elseif (currentIndex <= DBSIZE)
                        ReID_table(currentIndex).personObject.xPos = tmpPerson.xPos;
                        ReID_table(currentIndex).personObject.yPos = tmpPerson.yPos;
                        ReID_table(currentIndex).personObject.width = tmpPerson.width;
                        ReID_table(currentIndex).personObject.height = tmpPerson.height;
                        ReID_table(currentIndex).personObject.currentCamera = tmpPerson.currentCamera;
                        ReID_table(currentIndex).personObject.fv_array = tmpPerson.fv_array;
                        ReID_table(currentIndex).personObject.label = tmpPerson.label;
                        ReID_table(currentIndex).lru = 0;
                        currentIndex = currentIndex + 1;
                    else
                        for k = 1:DBSIZE
                            if (max < ReID_table(k).lru)
                                max = ReID_table(k).lru;
                                useIndex = k;
                            end
                        end
                        ReID_table(useIndex).personObject.xPos = tmpPerson.xPos;
                        ReID_table(useIndex).personObject.yPos = tmpPerson.yPos;
                        ReID_table(useIndex).personObject.width = tmpPerson.width;
                        ReID_table(useIndex).personObject.height = tmpPerson.height;
                        ReID_table(useIndex).personObject.currentCamera = tmpPerson.currentCamera;
                        ReID_table(useIndex).personObject.fv_array = tmpPerson.fv_array;
                        ReID_table(currentIndex).personObject.label = tmpPerson.label;
                        ReID_table(useIndex).lru = 0;
                    end
                end
            end
        end
        nodes(cam) = nodes(cam).fillRcvQ(sendQ);
    end
end
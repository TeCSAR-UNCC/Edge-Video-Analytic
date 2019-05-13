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
    edge_nodes(i) = EdgeNode(duke_cropped_camera_params(cameras(i),data_path, ...
                             mtmc_gt, camera_ranges(i,1), ...
                             camera_ranges(i,2), ...
                             60,5,200,20,0.5,12));
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
%%

for n = 1:length(cameras)
    [instances, avg_instances, miss_rates, avg_miss, id_modes, recall_precision, avg_recall, id_recall_recision, avg_id_recall, num_ids, avg_num_ids] = validation_stats(edge_nodes(n));
end

function r = duke_camera_params(id, dataPath, ground_truth, startFrame, endFrame, srcFrameRt, outFrameRt, tabSize, tabLife, kpcth, kpcnt)
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
        if ~isempty(ground_truth)
            gndtr = ground_truth(ground_truth(:,1)==id,2:end);
            gt_rows = sum((gndtr(:,2) == frames),2);
            gt = gndtr(gt_rows==1,:);
            [~,idx] = sort(gt(:,2));
            gt = gt(idx,:);
            num_ids = max(ground_truth(:,2));
        else
            gt = [];
            num_ids = 0;
        end
    else
        id = 0;
        cam_dets = zeros(1,76);
        cam_feats = zeros(1,1281);
        srcFrameRt = 0;
        outFrameRt = 1;
        frames=0:1;
        tabSize = 1;
        kpcth = 0;
        kpcnt = 0;
        tabLife = 1;
        gt = [];
        num_ids = 0;
    end
    
    r = struct('id',id,'dets',cam_dets,'feats',cam_feats,'sourceFrameRate',srcFrameRt, ...
               'outFrameRate',outFrameRt,'frames',frames,'gt',gt, ...
               'num_ids',num_ids,'tab_size',tabSize,'kpcth',kpcth, ...
               'kpcnt',kpcnt,'tabLife',tabLife);
end

function r = duke_cropped_camera_params(id, dataPath, ground_truth, startFrame, endFrame, srcFrameRt, outFrameRt, tabSize, tabLife, kpcth, kpcnt)
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
        feats = load(strcat(dataPath,'/features/camera',int2str(id),'cropped.mat'));
        cam_feats = feats.reid_features(test_rows==1,:);
        clear test_rows feats;
        if ~isempty(ground_truth)
            gndtr = ground_truth(ground_truth(:,1)==id,2:end);
            gt_rows = sum((gndtr(:,2) == frames),2);
            gt = gndtr(gt_rows==1,:);
            [~,idx] = sort(gt(:,2));
            gt = gt(idx,:);
            num_ids = max(ground_truth(:,2));
        else
            gt = [];
            num_ids = 0;
        end
    else
        id = 0;
        cam_dets = zeros(1,76);
        cam_feats = zeros(1,1281);
        srcFrameRt = 0;
        outFrameRt = 1;
        frames=0:1;
        tabSize = 1;
        kpcth = 0;
        kpcnt = 0;
        tabLife = 1;
        gt = [];
        num_ids = 0;
    end
    
    r = struct('id',id,'dets',cam_dets,'feats',cam_feats,'sourceFrameRate',srcFrameRt, ...
               'outFrameRate',outFrameRt,'frames',frames,'gt',gt, ...
               'num_ids',num_ids,'tab_size',tabSize,'kpcth',kpcth, ...
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

function [instances, avg_instances, miss_rates, avg_miss, id_modes, recall_precision, avg_recall, id_recall_precision, avg_id_recall, num_ids, avg_num_ids] = validation_stats(node)
    val_tab = node.val_table;
    
    val_len = size(val_tab,1);
    
    instances = zeros(val_len,1);
    miss_rates = zeros(val_len,1);
    id_modes = zeros(val_len,1);
    recall_precision = zeros(val_len,1);
    id_recall_precision = zeros(val_len,1);
    num_ids = zeros(val_len,1);
    successes = 0;
    
    valid_idxs = [];
    for i = 1:val_len
        if nnz(val_tab(i,:)) > 0
            instances(i) = nnz(val_tab(i,:));
            miss_rates(i) = length(find(val_tab(i,:)==-1))/nnz(val_tab(i,:));
            id_modes(i) = mode(val_tab(i,find(val_tab(i,:)>0)));
            num_ids(i) = length(unique(val_tab(i,find(val_tab(i,:)>0))));
            if (id_modes(i) > 0)
                recall_precision(i) = length(find(val_tab(i,:)==id_modes(i)))/nnz(val_tab(i,:));
                id_recall_precision(i) = recall_precision(i)/(1-miss_rates(i));
                successes = successes + length(find(val_tab(i,:)==id_modes(i)));
            end
            valid_idxs = [valid_idxs; i];
        end
    end
    
    avg_instances = mean(instances(valid_idxs));
    avg_miss = length(find(val_tab==-1))/nnz(val_tab);
    avg_recall = successes/nnz(val_tab);
    avg_id_recall = avg_recall/(1-avg_miss);
    avg_num_ids = mean(num_ids(valid_idxs));
end
clear
clc

fullValidation = 1;
cameras = [5];

load('data/ground_truth/mtmc_gt.mat');
if (fullValidation == 1)
    for cam = cameras
        cam_gt = mtmc_gt(mtmc_gt(:,1)==cam,:);
        camera_ranges(cam,:) = [min(cam_gt(:,3)), max(cam_gt(:,3))];
    end
    test_range = [min(camera_ranges(:,1)), max(camera_ranges(:,2))];
    clear cam_gt;
else
    for cam = cameras
        camera_ranges(cam,:) = [100000, 110000];
    end
end
clear cam;

data_path = 'data/DukeMTMC/detections';

cam5 = EdgeNode(camera_params(5,data_path,camera_ranges(5,1),camera_ranges(5,2),60,5,200,10,0.5,15));
%%

cam5 = cam5.resetNode();

for i = test_range(1):test_range(2)
    if (cam5.ready(i)==1)
        cam5 = cam5.process_step();
    end
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

    frame_interval = ceil(srcFrameRt/outFrameRt);
    frames = startFrame:frame_interval:endFrame;
    dets = load(strcat(dataPath,'/tecsar/camera',int2str(id),'.mat'));
    test_rows = sum((dets.detections(:,1) == frames),2);
    cam_dets = dets.detections(test_rows==1,:);
    clear dets;
    feats = load(strcat(dataPath,'/features/camera',int2str(id),'.mat'));
    cam_feats = feats.reid_features(test_rows==1,:);
    clear test_rows feats;
    
    r = struct('id',id,'dets',cam_dets,'feats',cam_feats,'sourceFrameRate',srcFrameRt, ...
               'outFrameRate',outFrameRt,'startFrame',startFrame, ...
               'endFrame',endFrame,'tab_size',tabSize,'kpcth',kpcth, ...
               'kpcnt',kpcnt,'tabLife',tabLife);
end

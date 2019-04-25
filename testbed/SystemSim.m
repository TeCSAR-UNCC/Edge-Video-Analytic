clear
clc

cam5 = EdgeNode(camera_params(5,1,60,60,5,200));

function r = camera_params(id, startFrame, endFrame, srcFrameRt, outFrameRt, tabSize)
    frame_interval = ceil((endFrame-startFrame+1)/outFrameRt);
    frames = startFrame:frame_interval:endFrame;
    dets = load(strcat('data/DukeMTMC/detections/tecsar/camera',int2str(id),'.mat'));
    test_rows = sum((dets.detections(:,1) == frames),2);
    cam_dets = dets.detections(test_rows==1,:);
    clear dets;
    feats = load(strcat('data/DukeMTMC/detections/features/camera',int2str(id),'.mat'));
    cam_feats = feats.reid_features(test_rows==1,:);
    clear test_rows feats;
    
    r = struct('dets',cam_dets,'feats',cam_feats,'sourceFrameRate',srcFrameRt, ...
               'outFrameRate',outFrameRt,'startFrame',startFrame, ...
               'endFrame',endFrame,'tab_size',tabSize);
end

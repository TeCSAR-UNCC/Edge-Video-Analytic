classdef EdgeNode
    properties
        id
        detections
        reid_feats
        currFrame
        startFrame
        endFrame
        sourceFrameRate
        frameRate
        currID
        
        kp_conf_thresh
        kp_count_thresh
        
        obj_table
        tab_size
        tab_life
    end
    methods
        function obj = EdgeNode(params)
            if nargin == 1
                obj.id = params.id;
                obj.detections = params.dets;
                obj.reid_feats = params.feats;
                obj.sourceFrameRate = params.sourceFrameRate;
                obj.startFrame = params.startFrame;
                obj.endFrame = params.endFrame;
                obj.currFrame = params.startFrame;
                obj.frameRate = params.outFrameRate;
                obj.kp_conf_thresh = params.kpcth;
                obj.kp_count_thresh = params.kpcnt;
                obj.currID = obj.id * 1000000;

                pt = personType(0,0,zeros(1,1280),0,0,0,0);
                oh = object_history(0,pt,0,0,0);
                obj.obj_table = repmat(oh,params.tab_size,1);
                obj.tab_size = params.tab_size;
                obj.tab_life = params.tabLife;
            else
                error('Expecting 1 argument of type camera_params');
            end
        end
        function obj = process_step(obj)
            [keypoints, validKPCounts, features, bboxes] = reid_inference(obj);
            currFrme = obj.currFrame;
            % Update table life vals. Send expired entries to server if previously
            %   sent to server.
            table_idxs = zeros(size(obj.obj_table,1),1);
            for i = 1:size(obj.obj_table,1)
                if (obj.obj_table(i).life > 0)
                    table_idxs(i) = 1;
                    obj.obj_table(i).life = obj.obj_table(i).life - 1;
                    if ((obj.obj_table(i).life==0) && (obj.obj_table(i).sentToServer==1))
                        obj.obj_table(i).sendObject.currentCamera = -1;
                        % Send to server to be implemented in the future
                    end
                end
            end
            % Server ReID Receive code to be implemented in the future

            % Local ReID
            dets = ones(size(keypoints,1),1);
            valid_dets = find(dets);
            valid_tabs = find(table_idxs);
            while (~isempty(valid_tabs) && ~isempty(valid_dets))
                match_table = Inf(size(keypoints,1),obj.tab_size);
                valid_tabs = find(table_idxs);
                valid_dets = find(dets);
                for tab = 1:length(valid_tabs)
                    for det = 1:length(valid_dets)
                        match_table(valid_dets(det),valid_tabs(tab)) = ...
                            0.85*norm(obj.obj_table(valid_tabs(tab)).sendObject.fv_array - ...
                                 features(valid_dets(det),:));
                        tab_person = obj.obj_table(valid_tabs(tab)).sendObject;
                        tab_bbox = [tab_person.xPos,tab_person.yPos,tab_person.width,tab_person.height];
                        match_table(valid_dets(det),valid_tabs(tab)) = ...
                            match_table(valid_dets(det),valid_tabs(tab)) + ...
                            0.15*(1 - iou(bboxes(valid_dets(det),:),tab_bbox));
                    end
                end
                best_match = min(match_table,[],'all');
                if (best_match < 0.6)
                    [det, tab] = find(match_table==best_match);
                    det = det(1);
                    tab = tab(1);
                    dets(det) = 0;
                    table_idxs(tab) = 0;
                    match_table(det,tab) = Inf;
                    if (validKPCounts(det) > obj.obj_table(tab).keyCount)
                        so = obj.obj_table(tab).sendObject;
                        so.fv_array = features(det);
                        so.xPos = bboxes(det,1);
                        so.yPos = bboxes(det,2);
                        so.width = bboxes(det,3);
                        so.height = bboxes(det,4);
                        obj.obj_table(tab).sendObject = so;
                        obj.obj_table(tab).life = obj.tab_life;
                        obj.obj_table(tab).keyCount = validKPCounts(det);
                    end
                else
                    valid_tabs = [];
                end
            end
            valid_dets = find(dets);
            if ~isempty(valid_dets)
                for i = 1:length(valid_dets)
                    sendObject = personType(obj.id,obj.currID,features(dets(valid_dets(i)),:), ...
                                            bboxes(dets(valid_dets(i)),1),bboxes(dets(valid_dets(i)),2), ...
                                            bboxes(dets(valid_dets(i)),3),bboxes(dets(valid_dets(i)),4));
                    obj.currID = obj.currID + 1;
                    if (obj.currID == (obj.id+1)*1000000)
                        obj.currID = obj.id*1000000;
                    end
                    firstOpenTab = Inf;
                    for tab = 1:obj.tab_size
                        if (obj.obj_table(tab).life == 0)
                            firstOpenTab = tab;
                            break;
                        end
                    end
                    obj.obj_table(firstOpenTab) = object_history(obj.tab_life, ...
                                                      sendObject, ...
                                                      validKPCounts(dets(valid_dets(i))), ...
                                                      0, 0);
                end
            end
            obj.currFrame =  obj.currFrame + ceil(obj.sourceFrameRate/obj.frameRate);
        end
        function r = ready(obj, frame)
            if (frame == obj.currFrame)
                r = 1;
            else
                r = 0;
            end
        end
        function obj = resetNode(obj)
            obj.currFrame = obj.startFrame;
            pt = personType(0,0,zeros(1,1280),0,0,0,0);
            oh = object_history(0,pt,0,0,0);
            obj.obj_table = repmat(oh,obj.tab_size,1);
            obj.currID = obj.id*1000000;
        end
    end
end

function r = personType(currentCamera,label,fv_array,xPos,yPos,width,height)
% PERSONTYPE Generates a personType struct
%   Inputs:
%       currentCamera - Camera ID
%       label - Detection ID label
%       fv_array - Encoded feature vector of the ID
%       xPos - X position of the target
%       yPos - Y position of the target
%       height - Height of the target
%       width - Width of the target
%   Output:
%       r - personType struct
    r = struct('currentCamera',int32(currentCamera),'label',int32(label), ...
               'fv_array',single(fv_array),'xPos',single(xPos),'yPos', ...
               single(yPos),'height',single(height),'width',single(width));
end

function r = object_history(life,sendObject,keyCount,reIDFlag,sentToServer)
% OBJECT_HISTORY Function for generating object_history struct
%   Inputs:
%       life - Remaining time a target will remain in the table
%       sendObject - personType struct for the target
%       keyCount - The number of valid keypoints for the target
%       reIDFlag - Whether or not the target has been reidentified by the
%           server
%       sentToServer - Whether or not the target has been sent to the
%           server for reid
%   Output:
%       r - object_history struct
    r = struct('life',int32(life),'sendObject',sendObject, ...
               'keyCount',int32(keyCount),'reIDFlag',int32(reIDFlag), ...
               'sentToServer',int32(sentToServer));
end

function [frameDets, keyCount, frameFeats, bboxes] = reid_inference(obj)
% REID_INFERENCE Finds the valid detections for the current frame and
% returns the corresponding keypoints, valid keypoint counts, ReID
% features, and bounding boxes
%   Inputs:
%       obj - The calling edge node
%   Outputs:
%       frameDets - Flattened keypoints for valid detections
%       keyCount - Number of valid keypoints for each valid detection
%       frameFeats - Encoded feature maps for valid detections
%       bboxs - Bounding boxes for valid detections

    frameIdxs = obj.detections(:,1)==obj.currFrame;
    % Detections for just the current frame
    frameDets = obj.detections(frameIdxs,2:end);
    % Features for just the current frame
    frameFeats = obj.reid_feats(frameIdxs,2:end);
    % Helper arrays
    valid_bboxes = zeros(size(frameDets,1),1);
    bboxes = zeros(size(frameDets,1),4);
    keyCount = zeros(size(frameDets,1),1);
    % Iterate over detections to find valid detections valid iff there are 
    %   at least kp_count_thresh keypoints with a confidence >= kp_conf_thresh,
    %   AND the bounding box associated with those keypoints has a nonzero area
    for det = 1:size(frameDets,1)
        % Reshape flat keypoints to [X,Y,Conf]
        keypoints = reshape(frameDets(det,:),3,25)';
        keypoint_conf = keypoints(:,3);
        % Extract keypoints with confidence >= kp_conf_thresh
        valid_mask = keypoint_conf >= obj.kp_conf_thresh;
        valid_keypoints = keypoints(valid_mask,1:2);
        % Count the number of valid keypoints
        key_cnt = nnz(valid_mask);
        if (key_cnt >= obj.kp_count_thresh)
            % If enough valid keypoints, extract bbox dims
            %   [min_x,min_y,max_x,max_y] and ensure they are in
            %   image bounds
            min_x = min(max(min(valid_keypoints(:,1)),0),1);
            min_y = min(max(min(valid_keypoints(:,2)),0),1);
            max_x = min(max(max(valid_keypoints(:,1)),0),1);
            max_y = min(max(max(valid_keypoints(:,2)),0),1);
            % Find width, height, and area
            width = max_x-min_x;
            height = max_y-min_y;
            area = (width)*(height);
            if (area > 0)
                % If area > 0 then the detection is valid
                bboxes(det,:) = [min_x,min_y,width,height];
                valid_bboxes(det) = 1;
                keyCount(det) = key_cnt;
            else
                valid_bboxes(det) = 0;
            end
        else
            valid_bboxes(det) = 0;
        end
    end
    % Reduce frameDets, frameFeats, bboxes, and keyCount to only
    %   valid instances
    valid_det = find(valid_bboxes);
    frameDets = frameDets(valid_det,:);
    frameFeats = frameFeats(valid_det,:);
    bboxes = bboxes(valid_det,:);
    keyCount = keyCount(valid_det);
end

function r = iou(box1,box2)
 % IOU Function for finding the Intersection over Union (IoU) for two boxes
 %  Inputs:
 %      box1 - First box (format: [x,y,width,height])
 %      box2 - Second box (format: [x,y,width,height])
 %  Output:
 %      r - IoU of the two boxes
 
    % box1 params
    minx1 = box1(1);
    maxx1 = box1(1)+box1(3);
    miny1 = box1(2);
    maxy1 = box1(2)+box1(4);
    % box2 params
    minx2 = box2(1);
    maxx2 = box2(1)+box2(3);
    miny2 = box2(2);
    maxy2 = box2(2)+box2(4);
    
    if (minx1 > maxx2 || maxx1 < minx2 || miny1 > maxy2 || maxy1 < miny2) 
        r = 0;
    else
        dx = min(maxx2, maxx1) - max(minx2, minx1);
        dy = min(maxy2, maxy1) - max(miny2, miny1);
        area1 = box1(3) * box1(4);
        area2 = box2(3) * box2(4);
        intersection = dx * dy;
        union = area1 + area2 - intersection;
        r = intersection/union;
    end
end
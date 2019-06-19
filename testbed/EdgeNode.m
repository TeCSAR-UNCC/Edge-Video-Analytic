classdef EdgeNode
    properties
        % Data structures
        detections % Detection trace for node
        reid_feats % Feature map trace for node
        obj_table % Local ReID table
        rcvQ % Receive queue from server
        sendQ % Send queue to server
        
        % Edge node intrinsic parameters
        id % Node ID number
        currFrame % Current frame number
        frames % Frames on which the edge node will operate
        currFrameIdx % Index of current frame in frames array
        sourceFrameRate % Frame rate of original capture
        frameRate % Operational frame rate of edge node
        currID % Label number for next new detection
        
        % Keypoint validation params
        kp_conf_thresh % Minimum confidence for valid keypoints
        kp_count_thresh % Minimum number of valid keypoints to generate a bounding box
        srv_kp_thresh
        
        % Table match params
        iou_weight % Weight of IoU in matching decision
        l2_weight % Weight of l2norm in matching decision
        match_threshold % Maximum score for positive match
        
        % Table params
        tab_size % Size of the local ReID table
        tab_life % Number of frames an ID is valid in ReID table
        max_tab_idx
        
        % Validation params
        gt % Ground truth data for edge node
        gt_frames
        num_ids % Number of unique IDs in the ground truth
        val_table % ID validation table
        gt_idx
    end
    methods
        function obj = EdgeNode(params)
            if nargin == 1
                obj.id = params.id;
                obj.detections = params.dets;
                obj.reid_feats = params.feats;
                obj.sourceFrameRate = params.sourceFrameRate;
                obj.frames = params.frames;
                obj.currFrameIdx = 1;
                obj.currFrame = obj.frames(obj.currFrameIdx);
                obj.frameRate = params.outFrameRate;
                obj.currID = obj.id * 1000000;
                
                obj.kp_conf_thresh = params.kpcth;
                obj.kp_count_thresh = params.kpcnt;
                
                obj.iou_weight = 0.15;
                obj.l2_weight = 0.85;
                obj.match_threshold = 0.8;

                obj.rcvQ = [];
                obj.sendQ = [];
                
                pt = personType(0,0,zeros(1,1280),0,0,0,0);
                oh = object_history(0,pt,0,0,0);
                obj.obj_table = repmat(oh,params.tab_size,1);
                obj.tab_size = params.tab_size;
                obj.tab_life = params.tabLife;
                obj.max_tab_idx = 1;
                
                obj.srv_kp_thresh = 20;
                
                obj.gt = params.gt;
                obj.gt_frames = obj.gt(:,2);
                obj.num_ids = params.num_ids;
                if obj.num_ids > 0
                    obj.val_table = zeros(obj.num_ids,length(obj.frames),'int32');
                end
                obj.gt_idx = 1;
            else
                error('Expecting 1 argument of type camera_params');
            end
        end
        function obj = process_step(obj)
            obj.sendQ = [];
            [keypoints, validKPCounts, features, bboxes] = reid_inference(obj);

            % Update table life vals. Send expired entries to server if previously
            %   sent to server.
            table_idxs = zeros(size(obj.obj_table,1),1);
            for i = 1:size(obj.obj_table,1)
                if (obj.obj_table(i).life > 0)
                    table_idxs(i) = 1;
                    obj.obj_table(i).life = obj.obj_table(i).life - 1;
                    if ((obj.obj_table(i).life==0) && (obj.obj_table(i).sentToServer==1))
                        obj.obj_table(i).sendObject.currentCamera = -1;
                        % Send expired labels to server if previously sent
                        obj.sendQ = [obj.sendQ; obj.obj_table(i).sendObject];
                    end
                end
            end
            % Check rcvQ for updates from server
            if (~isempty(obj.rcvQ))
                for q = 1:length(obj.rcvQ)
                    oldID = obj.rcvQ(q).oldID;
                    newID = obj.rcvQ(q).newID;
                    if obj.num_ids > 0
                        val_idx = find(obj.val_table(:,obj.currFrameIdx-1)==oldID);
                        if ~isempty(val_idx)
                            obj.val_table(val_idx(1),obj.currFrameIdx-1) = newID;
                        end
                    end
                    for i = 1:length(table_idxs)
                        if (obj.obj_table(i).sendObject.label == oldID)
                            obj.obj_table(i).sendObject.label = newID;
                            obj.obj_table(i).reIDFlag = 1;
                            break;
                        end
                    end
                end
            end
            obj.rcvQ = [];
            
            % Local ReID
            labeled_dets = [];

            dets = ones(size(keypoints,1),1);
            valid_dets = find(dets, 1);
            valid_tabs = find(table_idxs, 1);
            if (~isempty(valid_tabs) && ~isempty(valid_dets))
                match_table = Inf(size(keypoints,1),obj.max_tab_idx);
                valid_tabs = find(table_idxs);
                valid_dets = find(dets);
                for tab = 1:length(valid_tabs)
                    for det = 1:length(valid_dets)
                        match = ...
                            obj.l2_weight*vecnorm(obj.obj_table(valid_tabs(tab)).sendObject.fv_array - ...
                                 features(valid_dets(det),:),2,2);
                        tab_person = obj.obj_table(valid_tabs(tab)).sendObject;
                        tab_bbox = [tab_person.xPos,tab_person.yPos,tab_person.width,tab_person.height];
                        if (iou(bboxes(valid_dets(det),:),tab_bbox) == 0) && (match > 2)
                            match = Inf;
                        end
%                         match = match + obj.iou_weight*(1 - iou(bboxes(valid_dets(det),:),tab_bbox));
                        if (match < obj.match_threshold)
                            match_table(valid_dets(det),valid_tabs(tab)) = match;
                        end
                    end
                end
                best_match = min(match_table,[],'all');
                while (best_match < obj.match_threshold)
                    [det, tab] = find(match_table==best_match);
                    det = det(1);
                    tab = tab(1);
                    dets(det) = 0;
                    table_idxs(tab) = 0;
                    match_table(:,tab) = Inf;
                    match_table(det,:) = Inf;
                    so = obj.obj_table(tab).sendObject;
                    so.xPos = bboxes(det,1);
                    so.yPos = bboxes(det,2);
                    so.width = bboxes(det,3);
                    so.height = bboxes(det,4);
                    if (validKPCounts(det) > obj.obj_table(tab).keyCount)
                        so.fv_array = features(det,:);
                        obj.obj_table(tab).keyCount = validKPCounts(det);
                        if (validKPCounts(det) > obj.srv_kp_thresh)
                            obj.sendQ = [obj.sendQ; so];
                            obj.obj_table(tab).sentToServer = 1;
                        end
                    end
                    obj.obj_table(tab).sendObject = so;
                    obj.obj_table(tab).life = obj.tab_life;
                    
                    labeled_dets = [labeled_dets; [double(so.label), so.xPos, so.yPos, so.width, so.height]];
                    
                    best_match = min(match_table,[],'all');
                end
            end
            valid_dets = find(dets);
            if ~isempty(valid_dets)
                for i = 1:length(valid_dets)
                    sendObject = personType(obj.id,obj.currID,features(valid_dets(i),:), ...
                                            bboxes(valid_dets(i),1),bboxes(valid_dets(i),2), ...
                                            bboxes(valid_dets(i),3),bboxes(valid_dets(i),4));
                    so = sendObject;
                    labeled_dets = [labeled_dets; [double(so.label), so.xPos, so.yPos, so.width, so.height]];
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
                    if (validKPCounts(dets(valid_dets(i))) > obj.srv_kp_thresh)
                        obj.sendQ = [obj.sendQ; sendObject];
                        obj.obj_table(firstOpenTab).sentToServer = 1;
                    end
                    if (firstOpenTab > obj.max_tab_idx)
                        obj.max_tab_idx = firstOpenTab;
                    end
                end
            end
            
            if obj.num_ids > 0
%                 frame_gt = [];
%                 while (obj.gt_idx<=size(obj.gt,1)) && (obj.gt(obj.gt_idx,2)==obj.currFrame)
%                     frame_gt = [frame_gt; obj.gt(obj.gt_idx,:)];
%                     obj.gt_idx = obj.gt_idx + 1;
%                 end
                gt_idxs = find(obj.gt_frames==obj.currFrame);
                if ~isempty(gt_idxs)
                    frame_gt = obj.gt(gt_idxs,:);
                    obj.val_table(:,obj.currFrameIdx) = ...
                        gt_matching(obj.id, obj.val_table(:,obj.currFrameIdx), frame_gt, labeled_dets, obj.currFrame);
                end
            end
            
            obj.currFrameIdx =  obj.currFrameIdx + 1;
            if obj.currFrameIdx <= length(obj.frames)
                obj.currFrame = obj.frames(obj.currFrameIdx);
            end
        end
        function r = ready(obj, frame)
            if (frame == obj.currFrame)
                r = 1;
            else
                r = 0;
            end
        end
        function obj = resetNode(obj)
            obj.currFrameIdx = 1;
            obj.currFrame = obj.frames(obj.currFrameIdx);
            pt = personType(0,0,zeros(1,1280),0,0,0,0);
            oh = object_history(0,pt,0,0,0);
            obj.obj_table = repmat(oh,obj.tab_size,1);
            obj.currID = obj.id*1000000;
            obj.max_tab_idx = 1;
            obj.rcvQ = [];
            if obj.num_ids > 0
                obj.val_table = zeros(obj.num_ids,length(obj.frames),'int32');
            end
            obj.gt_idx = 1;
        end
        function obj = fillRcvQ(obj, queue)
            obj.rcvQ = queue;
        end
        function r = getSendQ(obj)
            r = obj.sendQ;
        end
        function obj = rstSendQ(obj)
            obj.sendQ = [];
        end
        function r = getID(obj)
            r = obj.id;
        end
        function obj = setMatchThresholdWeights(obj, l2w, thr)
            obj.iou_weight = 1-l2w;
            obj.l2_weight = l2w;
            obj.match_threshold = thr;
        end
    end
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
    if nargin ~= 5
        life = 0;
        sendObject = personType();
        keyCount = 0;
        reIDFlag = 0;
        sentToServer = 0;
    end
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
    body_head = [1:3,6,16:19]';
    body_torso = [2:10,13]';
    body_legs = [11,12,14,15,20:25]';

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
        % Count the number of valid keypoints
        key_cnt = nnz(valid_mask);
        key_cnt_head = nnz(valid_mask(body_head));
        key_cnt_torso = nnz(valid_mask(body_torso));
        key_cnt_legs = nnz(valid_mask(body_legs));
        % Extract keypoints with confidence >= 0.05
        valid_mask = keypoint_conf >= 0.05;
        valid_keypoints = keypoints(valid_mask,1:2);
%         if (key_cnt >= obj.kp_count_thresh)
        if (key_cnt_head > 0) && (key_cnt_torso > 1) && (key_cnt_legs > 0)
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

function r = gt_matching(id, validation, gt, dets, frame)  
    num_dets = size(dets,1);
    num_gt = size(gt,1);
    
    gt = single(gt);
    
    gt_cleared = ones(num_gt,1);
    
    matches = zeros(num_dets,num_gt);
    for d = 1:num_dets
        for g = 1:num_gt
            in = iou(dets(d,2:end), gt(g,3:end));

            if in > 0.3
                matches(d,g) = in;
            end
        end
    end
    
    valid = validation;
    
    if nnz(matches) < length(gt_cleared)
        im_name = sprintf('data/DukeMTMC/frames/camera%d/%06d.jpg',id,frame);
        save_name = sprintf('data/badframes/camera%d/%06d.jpg',id,frame);
        img = imread(im_name);
        for d = 1:num_dets
            scaleddet = dets(d,2:5).*[1920,1080,1920,1080];
            img = insertShape(img,'rectangle',scaleddet,'Color',[128+10*d,0,0],'LineWidth',3);
        end
        for g = 1:num_gt
            scaledgt = gt(g,3:6).*[1920,1080,1920,1080];
            img = insertShape(img,'rectangle',scaledgt,'Color',[0,128+10*g,0],'LineWidth',3);
        end
        imwrite(img,save_name);
    end
    
    while nnz(matches) > 0
        best_match = max(matches,[],'all');
        [d,g] = find(matches==best_match);
        valid(gt(g(1),1)) = dets(d(1),1);
        matches(d(1),:) = zeros(1,num_gt);
        matches(:,g(1)) = zeros(num_dets,1);
        gt_cleared(g(1)) = 0;
    end

    valid(gt(gt_cleared==1,1)) = -1;
    r = valid;
end
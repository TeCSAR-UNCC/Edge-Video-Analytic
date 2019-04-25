clear
clc

load('DukeMTMC/detections/tecsar/camera5.mat');

%%

num_detections = size(detections, 1);

full_bboxes = zeros(num_detections, 5);
cropped_bboxes = zeros(num_detections, 5);
cropped_keypoint_mask = [1,2,3,6,9,10,13,16,17,18,19];

for det = 1:num_detections
    frame = detections(det,1);
    detection = detections(det,2:end);
    
    % Full bounding box dimension calculations (head-to-toe)
    keypoints = reshape(detection,3,25)';
    keypoint_conf = keypoints(:,3);
    valid_mask = keypoint_conf >= 0.5;
    valid_keypoints = keypoints(valid_mask,1:2);
    if (~isempty(valid_keypoints))
        min_x = min(valid_keypoints(:,1));
        min_y = min(valid_keypoints(:,2));
        max_x = max(valid_keypoints(:,1));
        max_y = max(valid_keypoints(:,2));
        full_bboxes(det,:) = [frame min_x, min_y, max_x, max_y];
    else
        full_bboxes(det,:) = [frame 0 0 0 0];
    end
    
    % Cropped bounding box 
    keypoints = keypoints(cropped_keypoint_mask,:);
    keypoint_conf = keypoints(:,3);
    valid_mask = keypoint_conf >= 0.5;
    valid_keypoints = keypoints(valid_mask,1:2);
    if (~isempty(valid_keypoints))
        min_x = min(valid_keypoints(:,1));
        min_y = min(valid_keypoints(:,2));
        max_x = max(valid_keypoints(:,1));
        max_y = max(valid_keypoints(:,2));
        cropped_bboxes(det,:) = [frame min_x, min_y, max_x, max_y];
    else
        cropped_bboxes(det,:) = [frame 0 0 0 0];
    end
end

%%
bboxes = full_bboxes;
save('camera5_full_body.mat','bboxes');
bboxes = cropped_bboxes;
save('camera5_cropped_body.mat','bboxes');
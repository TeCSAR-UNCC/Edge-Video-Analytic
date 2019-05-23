function [] = ExtractBoundingBoxes(camera)
    dets = load(strcat('DukeMTMC/detections/tecsar/',camera,'.mat'));
    detections = dets.detections;
    num_detections = size(detections, 1);

    full_bboxes = zeros(num_detections, 5);
    % cropped_bboxes = zeros(num_detections, 5);
    % cropped_keypoint_mask = [1:4,6,7,9:11,13,14,16:19];
    fprintf('Extracting bounding boxes for %s\n',camera);
    for det = 1:num_detections
        frame = detections(det,1);
        detection = detections(det,2:end);

        % Full bounding box dimension calculations (head-to-toe)
        keypoints = reshape(detection,3,25)';
        keypoint_conf = keypoints(:,3);
        valid_mask = keypoint_conf >= 0.05;
        valid_keypoints = keypoints(valid_mask,1:2);
        if (~isempty(valid_keypoints))
            min_x = min(max(min(valid_keypoints(:,1)),0),1);
            min_y = min(max(min(valid_keypoints(:,2)),0),1);
            max_x = min(max(max(valid_keypoints(:,1)),0),1);
            max_y = min(max(max(valid_keypoints(:,2)),0),1);
            full_bboxes(det,:) = [frame min_x, min_y, max_x, max_y];
        else
            full_bboxes(det,:) = [frame 0 0 0 0];
        end

    %     % Cropped bounding box 
    %     keypoints = keypoints(cropped_keypoint_mask,:);
    %     keypoint_conf = keypoints(:,3);
    %     valid_mask = keypoint_conf >= 0.05;
    %     valid_keypoints = keypoints(valid_mask,1:2);
    %     if (~isempty(valid_keypoints))
    %         min_x = min(max(min(valid_keypoints(:,1)),0),1);
    %         min_y = min(max(min(valid_keypoints(:,2)),0),1);
    %         max_x = min(max(max(valid_keypoints(:,1)),0),1);
    %         max_y = min(max(max(valid_keypoints(:,2)),0),1);
    %         cropped_bboxes(det,:) = [frame min_x, min_y, max_x, max_y];
    %     else
    %         cropped_bboxes(det,:) = [frame 0 0 0 0];
    %     end
    end

    bboxes = full_bboxes;
    save(strcat('DukeMTMC/detections/bounding_boxes/',camera,'_full_body.mat'),'bboxes');
    % bboxes = cropped_bboxes;
    % save(strcat(camera,'_cropped_body.mat'),'bboxes');
    fprintf('Done\nBounding boxes written to: DukeMTMC/detections/bounding_boxes/%s_full_body.mat\n',camera);
end
clear
clc

files = dir('keypoints/*.json');

starting_frame = 1;
keypoints = zeros(10000000,76);
frame = starting_frame;
idx = 1;

for file = files'
    fid = fopen(strcat('keypoints/',file.name)); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid); 
    val = jsondecode(str);
    people = val.people;

    m = length(people);

    for i = 1:m
        new_keypoints = [frame, (people(i).pose_keypoints_2d(:))'];
        keypoints(idx,:) = new_keypoints;
        idx = idx + 1;
    end
    
    frame = frame + 1;
end

frameidxs = keypoints(:,1);
detections = keypoints(frameidxs ~= 0, :);
save camera5_detections.mat detections;

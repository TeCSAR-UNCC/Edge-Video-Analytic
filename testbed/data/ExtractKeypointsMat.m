function [] = ExtractKeypointsMat(camera)
    files = dir(strcat('keypoints/',camera,'/*.json'));

    start_frames = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];
    
    switch camera
        case 'camera1'
            starting_frame = start_frames(1);
        case 'camera2'
            starting_frame = start_frames(2);
        case 'camera3'
            starting_frame = start_frames(3);
        case 'camera4'
            starting_frame = start_frames(4);
        case 'camera5'
            starting_frame = start_frames(5);
        case 'camera6'
            starting_frame = start_frames(6);
        case 'camera7'
            starting_frame = start_frames(7);
        case 'camera8'
            starting_frame = start_frames(8);
    end
    keypoints = zeros(10000000,76);
    frame = starting_frame;
    idx = 1;
    frame_file = 1;

    for file = files'
        clc;
        fprintf('Extracting Keypoints:\nProcessing Keypoint JSON %d of %d\n',frame_file,length(files));
        fid = fopen(strcat('keypoints/',camera,'/',file.name)); 
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
        frame_file = frame_file+1;
    end

    frameidxs = keypoints(:,1);
    detections = keypoints(frameidxs ~= 0, :);
    save(strcat('DukeMTMC/detections/tecsar/',camera,'.mat'), 'detections');
    fprintf('Done\nKeypoints written to: DukeMTMC/detections/tecsar/%s.mat\n',camera);
    exit;
end

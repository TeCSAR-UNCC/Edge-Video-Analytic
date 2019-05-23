function [] = CombineFeatureMats(camera)
    det = load(strcat('DukeMTMC/detections/tecsar/',camera,'.mat'));
    detections = det.detections;
    files = dir(strcat('DukeMTMC/detections/features/',camera,'/*.mat'));

    reid_features = single(zeros(size(detections,1),1281));
    reid_features(:,1) = single(detections(:,1));

    clear detections

    idx = 1;
    file_num = 1;
    for file = files'
        clc;
        fprintf('Combining Feature mats:\nProcessing feature mat %d of %d\n',file_num,length(files));
        feats = load(strcat(file.folder,'/',file.name));
        features = feats.features;
        range = idx:(idx+size(features,1)-1);
        reid_features(range,2:end) = features;
        idx = range(end)+1;
        file_num = file_num + 1;
    end

    save(strcat('DukeMTMC/detections/features/',camera,'.mat'),'reid_features','-v7.3');
    fprintf('Done\nFeatures saved to: DukeMTMC/detections/features/%s.mat\n',camera);
end
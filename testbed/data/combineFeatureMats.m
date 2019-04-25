clear
clc

camera = 'camera5';

load(strcat('DukeMTMC/detections/tecsar/',camera,'.mat'));

files = dir(strcat('DukeMTMC/detections/features/',camera,'/*.mat'));

reid_features = single(zeros(size(detections,1),1281));
reid_features(:,1) = single(detections(:,1));

clear detections

idx = 1;

for file = files'
    load(strcat(file.folder,'/',file.name));
    range = idx:(idx+size(features,1)-1);
    reid_features(range,2:end) = features;
    idx = range(end)+1;
end

save(strcat('DukeMTMC/detections/features/',camera,'.mat'),'reid_features','-v7.3');
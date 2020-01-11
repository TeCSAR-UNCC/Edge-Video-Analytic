clear
clc

fullValidation = 0;
cameras = [5];
camera_ranges = zeros(length(cameras),2);
server = 1;
%127720:187540

load('../data/ground_truth/mtmc_gt.mat');
if (fullValidation == 1)
    for cam = cameras
        cam_gt = mtmc_gt(mtmc_gt(:,1)==cam,:);
        camera_ranges(cameras==cam,:) = [min(cam_gt(:,3)), max(cam_gt(:,3))];
    end
    test_range = [min(camera_ranges(:,1)), max(camera_ranges(:,2))];
    clear cam_gt;
else
    for cam = cameras
        cam_gt = mtmc_gt(mtmc_gt(:,1)==cam,:);
        camera_ranges(cameras==cam,:) = [127720, 187540];
    end
    test_range = [min(camera_ranges(:,1)), max(camera_ranges(:,2))];
    clear cam_gt;
end
clear cam;

%data_path = '/home/mmendiet/NVDA_DETEC/Edge-Video-Analytic/testbed/matlab_lstm_5fps_retry/mnt/home/mmendiet/Edge-Video-Analytic/testbed/data3/detections';
data_path = '/mnt/nas/Datasets/DukeMTMC/detections/tecsar';

for i = 1:length(cameras)
    edge_nodes(i) = EdgeNode(duke_camera_params(cameras(i),data_path, ...
                             mtmc_gt, camera_ranges(i,1), ...
                             camera_ranges(i,2), ...
                             60,5,200,6,0.5,5));%EDIT****Keypoints 5,8 ***EDIT frame rate 5 to 4
end
%%

for i = 1:length(cameras)
    edge_nodes(i) = edge_nodes(i).setMatchThresholdWeights(1,9999);
    edge_nodes(i) = edge_nodes(i).resetNode(); % Reset Edge Nodes
end

edge_server_ops(edge_nodes,1); % Reset Edge Server

progressbar('Simulation Status','Post-Analytics');
for i = test_range(1):test_range(2)
   camReady = 0;
    for n = 1:length(cameras)
        if (edge_nodes(n).ready(i)==1)
            edge_nodes(n) = edge_nodes(n).process_step();
            camReady = camReady+1;
            if(length(cameras)==1)
                edge_nodes(n) = edge_nodes(n).rstSendQ();
            end
        end
    end
     if(server==1)
         edge_nodes = edge_server_ops(edge_nodes,0);
     end
    progressbar((i-test_range(1))/(test_range(2)-test_range(1)),[]);
end
%%
if(length(cameras) >= 1)
    [instances, avg_instances, miss_rates, avg_miss, id_modes, id_mode_counts, ...
    recall_precision, avg_recall, id_recall_precision, avg_id_recall, ...
     num_ids, avg_num_ids, idtp,idfp,idfn,IDP,IDR,IDF1,...
     IDRD,IDF1D,IDPnode,IDRnode,IDF1node] = validation_statsAll(edge_nodes);
 
    % for cc = 1:length(cameras)
    %     [confusion_matrix] = confusion_analysis(edge_nodes(cc).val_table, id_modes);
    %     [sample_confusion_matrix] = sample_confusion_analysis(edge_nodes(cc).val_table, id_modes, 50);
    % end
    progressbar([],n/length(cameras));

    t = datetime('now','TimeZone','America/New_York','Format','d-MMM-y HH:mm:ss');
    savename = sprintf('results/multi/multi_%s.mat',t);
    save(savename, 'instances', 'avg_instances', 'miss_rates', 'avg_miss', 'id_modes', ...
        'id_mode_counts', 'recall_precision', 'avg_recall', 'id_recall_precision', ...
        'avg_id_recall', 'num_ids', 'avg_num_ids', ...
        'idtp','idfp','idfn','IDP','IDR','IDF1','IDRD','IDF1D','IDPnode','IDRnode','IDF1node');

else
    %[instances, avg_instances, miss_rates, avg_miss, id_modes, id_mode_counts, ...
    %     recall_precision, avg_recall, id_recall_precision, avg_id_recall, ...
     %    num_ids, avg_num_ids, idtp,idfp,idfn,IDP,IDR,IDF1] = validation_stats(edge_nodes(n));
     
    [instances, avg_instances, miss_rates, avg_miss, id_modes, id_mode_counts, ...
    recall_precision, avg_recall, id_recall_precision, avg_id_recall, ...
     num_ids, avg_num_ids, idtp,idfp,idfn,IDP,IDR,IDF1,...
     IDRD,IDF1D,IDPnode,IDRnode,IDF1node] = validation_statsAll(edge_nodes);

    [confusion_matrix] = confusion_analysis(edge_nodes(1).val_table, id_modes);
    [sample_confusion_matrix] = sample_confusion_analysis(edge_nodes(1).val_table, id_modes, 50);
    progressbar([],1/length(cameras));
        
%     t = datetime('now','TimeZone','America/New_York','Format','d-MMM-y HH:mm:ss');
%     save_name = sprintf('results/camera%d/conMat/camera%d_%s_Full.png',cameras(1),cameras(1),t);
%     saveas(confusion_matrix,save_name);
%     save_name = sprintf('results/camera%d/conMat/camera%d_%s_Sample.png',cameras(1),cameras(1),t);
%     imwrite(sample_confusion_matrix,save_name);
    
    
%     savename = sprintf('results/camera%d/camera%d_%s.mat',cameras(1),cameras(1),t);
%     save(savename, 'instances', 'avg_instances', 'miss_rates', 'avg_miss', 'id_modes', ...
%         'id_mode_counts', 'recall_precision', 'avg_recall', 'id_recall_precision', ...
%         'avg_id_recall', 'num_ids', 'avg_num_ids', 'confusion_matrix', ...
%         'idtp','idfp','idfn','IDP','IDR','IDF1');
    t = datetime('now','TimeZone','America/New_York','Format','d-MMM-y HH:mm:ss');
    %savename = sprintf('results/multi/multi_%s.mat',t);
    savename = sprintf('results/camera%d/camera%d_%s.mat',cameras(1),cameras(1),t);
    save(savename, 'instances', 'avg_instances', 'miss_rates', 'avg_miss', 'id_modes', ...
        'id_mode_counts', 'recall_precision', 'avg_recall', 'id_recall_precision', ...
        'avg_id_recall', 'num_ids', 'avg_num_ids', ...
        'idtp','idfp','idfn','IDP','IDR','IDF1','IDRD','IDF1D','IDPnode','IDRnode','IDF1node');

end
    

function r = duke_camera_params(id, dataPath, ground_truth, startFrame, endFrame, srcFrameRt, outFrameRt, tabSize, tabLife, kpcth, kpcnt)
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
start_frames = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766];
    if nargin > 1
        frame_interval = ceil(srcFrameRt/outFrameRt);
        %frames = ((startFrame-start_frames(id)+1):frame_interval:(endFrame-start_frames(id)+1));
        frames = (startFrame:frame_interval:endFrame);
        dets = load(strcat(dataPath,'/camera',int2str(id),'.mat'));
        test_rows = sum((dets.detections(:,1) == frames),2);
        cam_dets = dets.detections(test_rows==1,:);
        clear dets;
        feats = load(strcat('/mnt/nas/Datasets/DukeMTMC/detections/features/camera',int2str(id),'.mat'));
        cam_feats = feats.reid_features(test_rows==1,:);
        clear test_rows feats;
        if ~isempty(ground_truth)
            gndtr = ground_truth(ground_truth(:,1)==id,2:end);
            gndtr(:,2) = gndtr(:,2)+start_frames(id)-1;
            gt_rows = sum((gndtr(:,2) == frames),2);
            gt = gndtr(gt_rows==1,:);
            [~,idx] = sort(gt(:,2));
            gt = gt(idx,:);
            num_ids = max(ground_truth(:,2));
        else
            gt = [];
            num_ids = 0;
        end
    else
        id = 0;
        cam_dets = zeros(1,76);
        cam_feats = zeros(1,1281);
        srcFrameRt = 0;
        outFrameRt = 1;
        frames=0:1;
        tabSize = 1;
        kpcth = 0;
        kpcnt = 0;
        tabLife = 1;
        gt = [];
        num_ids = 0;
    end
    
    r = struct('id',id,'dets',cam_dets,'feats',cam_feats,'sourceFrameRate',srcFrameRt, ...
               'outFrameRate',outFrameRt,'frames',frames,'gt',gt, ...
               'num_ids',num_ids,'tab_size',tabSize,'kpcth',kpcth, ...
               'kpcnt',kpcnt,'tabLife',tabLife);
end

function r = duke_cropped_camera_params(id, dataPath, ground_truth, startFrame, endFrame, srcFrameRt, outFrameRt, tabSize, tabLife, kpcth, kpcnt)
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
    if nargin > 1
        frame_interval = ceil(srcFrameRt/outFrameRt);
        frames = (startFrame:frame_interval:endFrame);
        dets = load(strcat(dataPath,'/tecsar/camera',int2str(id),'.mat'));
        test_rows = sum((dets.detections(:,1) == frames),2);
        cam_dets = dets.detections(test_rows==1,:);
        clear dets;
        feats = load(strcat(dataPath,'/features/camera',int2str(id),'cropped.mat'));
        cam_feats = feats.reid_features(test_rows==1,:);
        clear test_rows feats;
        if ~isempty(ground_truth)
            gndtr = ground_truth(ground_truth(:,1)==id,2:end);
            gt_rows = sum((gndtr(:,2) == frames),2);
            gt = gndtr(gt_rows==1,:);
            [~,idx] = sort(gt(:,2));
            gt = gt(idx,:);
            num_ids = max(ground_truth(:,2));
        else
            gt = [];
            num_ids = 0;
        end
    else
        id = 0;
        cam_dets = zeros(1,76);
        cam_feats = zeros(1,1281);
        srcFrameRt = 0;
        outFrameRt = 1;
        frames=0:1;
        tabSize = 1;
        kpcth = 0;
        kpcnt = 0;
        tabLife = 1;
        gt = [];
        num_ids = 0;
    end
    
    r = struct('id',id,'dets',cam_dets,'feats',cam_feats,'sourceFrameRate',srcFrameRt, ...
               'outFrameRate',outFrameRt,'frames',frames,'gt',gt, ...
               'num_ids',num_ids,'tab_size',tabSize,'kpcth',kpcth, ...
               'kpcnt',kpcnt,'tabLife',tabLife);
end

function r = object_history(personObject, lru)
    r = struct('personObject',personObject,'lru',lru);
end

function nodes = edge_server_ops(nodes, rst)
    if nargin == 1
        rst = 0;
    elseif nargin == 0
        error('Expecting at least 1 argument');
    end

    persistent ReID_table;
    persistent currentIndex;
    
    DBSIZE = 1000;
    L2_THR = 1.5; %4.25 ****EDIT*****1.5
    
    if (isempty(ReID_table) || (rst==1))
        pt = personType();
        oh = object_history(pt, 1000);
        ReID_table = repmat(oh,DBSIZE,1);
        currentIndex = 1;
        return;
    end
    
    for i = 1:DBSIZE
        ReID_table(i).lru = ReID_table(i).lru + 1;
    end
    
    sendQ = [];
    
    for cam = 1:size(nodes,2)
        queue = nodes(cam).getSendQ();
    
        if (~isempty(queue))
            for q = 1:length(queue)
                tmpPerson = queue(q);
                matchIdx = -1;
                tmpmatchIdx = -1;
                minL2 = Inf;
                updateFlag = 0;
                max = -1;
                
                for i = 1:DBSIZE
                    ReID_person = ReID_table(i).personObject;
                    if (tmpPerson.currentCamera > -1)
                        match = norm(tmpPerson.fv_array - ReID_person.fv_array);
                        if ((ReID_person.label == tmpPerson.label) && ...
                            (ReID_person.currentCamera == tmpPerson.currentCamera))
                            updateFlag = 1;
                            matchIdx = i;
                            break;
                        elseif ((match < minL2) && ReID_person.currentCamera == -1)
                            minL2 = match;
                            tmpmatchIdx = i;
                        end
                    elseif (ReID_person.label == tmpPerson.label)
                        ReID_table(i).personObject.currentCamera = -1;
                        break;
                    end
                end

                if ((tmpPerson.currentCamera > -1))
                    if((minL2 < L2_THR) && (updateFlag == 0))
                       matchIdx = tmpmatchIdx; 
                    end
                    if (matchIdx > 0)
                        ReID_table(matchIdx).personObject.xPos = tmpPerson.xPos;
                        ReID_table(matchIdx).personObject.yPos = tmpPerson.yPos;
                        ReID_table(matchIdx).personObject.width = tmpPerson.width;
                        ReID_table(matchIdx).personObject.height = tmpPerson.height;
                        ReID_table(matchIdx).personObject.currentCamera = tmpPerson.currentCamera;
                        ReID_table(matchIdx).personObject.fv_array = tmpPerson.fv_array;
                        ReID_table(matchIdx).lru = 0;
                        sendQ = [sendQ; reIDType(tmpPerson.label,ReID_table(matchIdx).personObject.label)];
                    elseif (currentIndex <= DBSIZE)
                        ReID_table(currentIndex).personObject.xPos = tmpPerson.xPos;
                        ReID_table(currentIndex).personObject.yPos = tmpPerson.yPos;
                        ReID_table(currentIndex).personObject.width = tmpPerson.width;
                        ReID_table(currentIndex).personObject.height = tmpPerson.height;
                        ReID_table(currentIndex).personObject.currentCamera = tmpPerson.currentCamera;
                        ReID_table(currentIndex).personObject.fv_array = tmpPerson.fv_array;
                        ReID_table(currentIndex).personObject.label = tmpPerson.label;
                        ReID_table(currentIndex).lru = 0;
                        currentIndex = currentIndex + 1;
                    else
                        for k = 1:DBSIZE
                            if (max < ReID_table(k).lru)
                                max = ReID_table(k).lru;
                                useIndex = k;
                            end
                        end
                        ReID_table(useIndex).personObject.xPos = tmpPerson.xPos;
                        ReID_table(useIndex).personObject.yPos = tmpPerson.yPos;
                        ReID_table(useIndex).personObject.width = tmpPerson.width;
                        ReID_table(useIndex).personObject.height = tmpPerson.height;
                        ReID_table(useIndex).personObject.currentCamera = tmpPerson.currentCamera;
                        ReID_table(useIndex).personObject.fv_array = tmpPerson.fv_array;
                        ReID_table(currentIndex).personObject.label = tmpPerson.label;
                        ReID_table(useIndex).lru = 0;
                    end
                end
            end
        end
        nodes(cam) = nodes(cam).fillRcvQ(sendQ);
        nodes(cam) = nodes(cam).rstSendQ();
    end
end

function r = ranked_modes(vec)
    modes = [];
    mode_counts = [];
    mode_first_idxs = [];
    orig_vec = vec;
    vec = vec(vec>0);
    while ~isempty(vec)
        newMode = mode(vec);
        mode_idxs = find(orig_vec==newMode);
        modes = [modes, newMode];
        mode_counts = [mode_counts, length(mode_idxs)];
        mode_first_idxs = [mode_first_idxs, mode_idxs(1)];
        vec = vec(vec~=newMode);
    end
    
    r = struct('ModeList',modes,'ModeCounts',mode_counts,'ModeStarts',mode_first_idxs);
end

function [instances, avg_instances, miss_rates, avg_miss, ...
          id_modes, id_mode_counts, recall_precision, avg_recall, ...
          id_recall_precision, avg_id_recall, ...
          num_ids, avg_num_ids, idtp,idfp,idfn,IDP,IDR,IDF1] = validation_stats(node)
    val_tab = node.val_table;
    
    val_len = size(val_tab,1);
    
    instances = zeros(val_len,1);
    miss_rates = zeros(val_len,1);
    id_modes = zeros(val_len,1);
    id_mode_counts = zeros(val_len,1);
    id_ranked_modes = repmat(ranked_modes(0),val_len,1);
    recall_precision = zeros(val_len,1);
    id_recall_precision = zeros(val_len,1);
    num_ids = zeros(val_len,1);
    successes = 0;
    no_mode = -1;
    idtp = 0;
    idfp = 0;
    idfn = 0;
    IDP = 0;
    IDR = 0;
    IDF1 = 0;
    
    valid_idxs = [];
    for i = 1:val_len
        if nnz(val_tab(i,:)) > 0
            instances(i) = nnz(val_tab(i,:));
            miss_rates(i) = length(find(val_tab(i,:)==-1))/nnz(val_tab(i,:));
            nonzeros = val_tab(i,val_tab(i,:)>0);
            id_ranked_modes(i) = ranked_modes(val_tab(i,:));
            num_ids(i) = length(unique(nonzeros));
            valid_idxs = [valid_idxs; i];
        end
    end
    
    for i = 1:length(valid_idxs)
        idx = valid_idxs(i);
        mode_list = id_ranked_modes(idx).ModeList;
        mode_counts = id_ranked_modes(idx).ModeCounts;
        if ~isempty(mode_list)
            id_modes(idx) = mode_list(1);
            id_mode_counts(idx) = mode_counts(1);
        else
            id_modes(idx) = no_mode;
            id_mode_counts(idx) = 0;
            no_mode = no_mode - 1;
        end
    end
    shared_ids = [];
    unique_ids = unique(id_modes(id_modes>0));

    for i = 1:length(unique_ids)
        sharers = find(id_modes==unique_ids(i));
        if length(sharers) > 1
            shared_ids = [shared_ids; struct('Label',unique_ids(i),'Sharers',sharers)];
        end
    end
    while(~isempty(shared_ids))
        for i = 1:length(shared_ids)
            sharers = shared_ids(i).Sharers;
            starts = zeros(length(sharers),1);
            for j = 1:length(sharers)
                starts(j) = id_ranked_modes(sharers(j)).ModeStarts(1);
            end
            [~,first_start] = min(starts);
            toBeUpdated = find(starts~=first_start);
            for j = 1:length(toBeUpdated)
                id_ranked_modes(sharers(toBeUpdated(j))).ModeList(1) = [];
                id_ranked_modes(sharers(toBeUpdated(j))).ModeCounts(1) = [];
                id_ranked_modes(sharers(toBeUpdated(j))).ModeStarts(1) = [];
            end
        end
        for i = 1:length(valid_idxs)
            idx = valid_idxs(i);
            mode_list = id_ranked_modes(idx).ModeList;
            mode_counts = id_ranked_modes(idx).ModeCounts;
            if ~isempty(mode_list)
                id_modes(idx) = mode_list(1);
                id_mode_counts(idx) = mode_counts(1);
            else
                id_modes(idx) = no_mode;
                id_mode_counts(idx) = 0;
                no_mode = no_mode - 1;
            end
        end
        shared_ids = [];
        unique_ids = unique(id_modes(id_modes>0));

        for i = 1:length(unique_ids)
            sharers = find(id_modes==unique_ids(i));
            if length(sharers) > 1
                shared_ids = [shared_ids; struct('Label',unique_ids(i),'Sharers',sharers)];
            end
        end
    end
    
    for i = 1:val_len
        if nnz(val_tab(i,:)) > 0
            if (id_modes(i) > 0)
                recall_precision(i) = id_mode_counts(i)/nnz(val_tab(i,:));
                id_recall_precision(i) = recall_precision(i)/(1-miss_rates(i));
                successes = successes + length(find(val_tab(i,:)==id_modes(i)));
                idtp = id_mode_counts(i) + idtp;
                idfp = (nnz(val_tab==id_modes(i)) - id_mode_counts(i)) + idfp;
                idfn = (nnz(val_tab(i,:)) - id_mode_counts(i)) + idfn;
                IDP = idtp/(idtp+idfp);
                IDR = idtp/(idtp+idfn);
                IDF1 = (2*idtp)/((2*idtp)+idfp+idfn);
                
            end
        end
    end
    
    avg_instances = mean(instances(valid_idxs));
    avg_miss = length(find(val_tab==-1))/nnz(val_tab);
    avg_recall = successes/nnz(val_tab);
    avg_id_recall = avg_recall/(1-avg_miss);
    avg_num_ids = mean(num_ids(valid_idxs));
end

function [instances, avg_instances, miss_rates, avg_miss, ...
          id_modes, id_mode_counts, recall_precision, avg_recall, ...
          id_recall_precision, avg_id_recall, ...
          num_ids, avg_num_ids, idtp,idfp,idfn,IDP,IDR,IDF1 ...
          ,IDRD,IDF1D,IDPnode,IDRnode,IDF1node] = validation_statsAll(enodes)
    
    
    val_lenAll = zeros(1,size(enodes,2));
    val_tabAll = zeros(size(enodes(1).val_table,1),size(enodes(1).val_table,2),size(enodes,2));
    
    val_tab = enodes(1).val_table;
    val_lenTemp = size(val_tab,1);
    
    instances = zeros(val_lenTemp,size(enodes,2));
    miss_rates = zeros(val_lenTemp,size(enodes,2));
    id_modes = zeros(val_lenTemp,size(enodes,2));
    id_mode_counts = zeros(val_lenTemp,size(enodes,2));
    id_ranked_modes = repmat(ranked_modes(0),val_lenTemp,size(enodes,2));
    recall_precision = zeros(val_lenTemp,size(enodes,2));
    id_recall_precision = zeros(val_lenTemp,size(enodes,2));
    num_ids = zeros(val_lenTemp,size(enodes,2));
    successes = 0;
    no_mode = -1;
    idtp = 0;
    idfp = 0;
    idfn = 0;
    idfnD = 0;
    IDP = 0;
    IDR = 0;
    IDF1 = 0;
    IDRD = 0;
    IDF1D = 0;
    IDPnode = zeros(size(enodes,2),2);
    IDRnode = zeros(size(enodes,2),3);
    IDF1node = zeros(size(enodes,2),3);
    
    for i=1:size(enodes,2)
      val_tabAll(:,:,i) = enodes(i).val_table;
      val_lenAll(1,i) = size(val_tabAll(i),1);
    end
    
    for cam=1:size(enodes,2)
        val_tab = enodes(cam).val_table;
        val_len = size(val_tab,1);
        idtpN = 0;
        idfpN = 0;
        idfnN = 0;
        idfnND = 0;
        
        valid_idxs = [];
        for i = 1:val_len
            if nnz(val_tab(i,:)) > 0
                instances(i) = nnz(val_tab(i,:));
                miss_rates(i) = length(find(val_tab(i,:)==-1))/nnz(val_tab(i,:));
                nonzeros = val_tab(i,val_tab(i,:)>0);
                id_ranked_modes(i) = ranked_modes(val_tab(i,:));
                num_ids(i) = length(unique(nonzeros));
                valid_idxs = [valid_idxs; i];
            end
        end

        for i = 1:length(valid_idxs)
            idx = valid_idxs(i);
            mode_list = id_ranked_modes(idx).ModeList;
            mode_counts = id_ranked_modes(idx).ModeCounts;
            if ~isempty(mode_list)
                id_modes(idx) = mode_list(1);
                id_mode_counts(idx) = mode_counts(1);
            else
                id_modes(idx) = no_mode;
                id_mode_counts(idx) = 0;
                no_mode = no_mode - 1;
            end
        end
        shared_ids = [];
        unique_ids = unique(id_modes(id_modes>0));

        for i = 1:length(unique_ids)
            sharers = find(id_modes==unique_ids(i));
            if length(sharers) > 1
                shared_ids = [shared_ids; struct('Label',unique_ids(i),'Sharers',sharers)];
            end
        end
        while(~isempty(shared_ids))
            for i = 1:length(shared_ids)
                sharers = shared_ids(i).Sharers;
                starts = zeros(length(sharers),1);
                for j = 1:length(sharers)
                    if(~isempty(id_ranked_modes(sharers(j)).ModeStarts))
                        starts(j) = id_ranked_modes(sharers(j)).ModeStarts(1);
                    end
                end
                [~,first_start] = min(starts);
                toBeUpdated = find(starts~=first_start);
                for j = 1:length(toBeUpdated)
                    if(~isempty(id_ranked_modes(sharers(toBeUpdated(j))).ModeList))
                        id_ranked_modes(sharers(toBeUpdated(j))).ModeList(1) = [];
                    end
                    if(~isempty(id_ranked_modes(sharers(toBeUpdated(j))).ModeCounts))
                        id_ranked_modes(sharers(toBeUpdated(j))).ModeCounts(1) = [];
                    end
                    if(~isempty(id_ranked_modes(sharers(toBeUpdated(j))).ModeStarts))
                        id_ranked_modes(sharers(toBeUpdated(j))).ModeStarts(1) = [];
                    end
                end
            end
            for i = 1:length(valid_idxs)
                idx = valid_idxs(i);
                mode_list = id_ranked_modes(idx).ModeList;
                mode_counts = id_ranked_modes(idx).ModeCounts;
                if ~isempty(mode_list)
                    id_modes(idx) = mode_list(1);
                    id_mode_counts(idx) = mode_counts(1);
                else
                    id_modes(idx) = no_mode;
                    id_mode_counts(idx) = 0;
                    no_mode = no_mode - 1;
                end
            end
            shared_ids = [];
            unique_ids = unique(id_modes(id_modes>0));

            for i = 1:length(unique_ids)
                sharers = find(id_modes==unique_ids(i));
                if length(sharers) > 1
                    shared_ids = [shared_ids; struct('Label',unique_ids(i),'Sharers',sharers)];
                end
            end
        end

        for i = 1:val_len
            if nnz(val_tab(i,:)) > 0
                if (id_modes(i) > 0)
                    recall_precision(i) = id_mode_counts(i)/nnz(val_tab(i,:));
                    id_recall_precision(i) = recall_precision(i)/(1-miss_rates(i));
                    successes = successes + length(find(val_tab(i,:)==id_modes(i)));
                    idtp = id_mode_counts(i) + idtp;
                    idfp = (nnz(val_tabAll==id_modes(i)) - id_mode_counts(i)) + idfp;
                    idfn = (nnz(val_tab(i,:)) - id_mode_counts(i)) + idfn;
                    idfnD = (nnz(val_tab(i,:)) - nnz(val_tab(i,:)==-1) - id_mode_counts(i)) + idfnD;
                    
                    idtpN = id_mode_counts(i) + idtpN;
                    idfpN = (nnz(val_tabAll==id_modes(i)) - id_mode_counts(i)) + idfpN;
                    idfnN = (nnz(val_tab(i,:)) - id_mode_counts(i)) + idfnN;
                    idfnND = (nnz(val_tab(i,:)) - nnz(val_tab(i,:)==-1) - id_mode_counts(i)) + idfnND;
                end
            end
        end
        IDPnode(cam,1) = idtpN/(idtpN+idfpN);
        IDPnode(cam,2) = enodes(cam).id;
        IDRnode(cam,1) = idtpN/(idtpN+idfnN);
        IDRnode(cam,2) = idtpN/(idtpN+idfnND);
        IDRnode(cam,3) = enodes(cam).id;
        IDF1node(cam,1) = (2*idtpN)/((2*idtpN)+idfpN+idfnN);
        IDF1node(cam,2) = (2*idtpN)/((2*idtpN)+idfpN+idfnND);
        IDF1node(cam,3) = enodes(cam).id;
        avg_instances(cam,1) = mean(instances(valid_idxs));
        avg_miss(cam,1) = length(find(val_tab==-1))/nnz(val_tab);
        avg_recall(cam,1) = successes/nnz(val_tab);
        avg_id_recall(cam,1) = avg_recall(cam,1)/(1-avg_miss(cam,1));
        avg_num_ids(cam,1) = mean(num_ids(valid_idxs));
    end
    IDP = idtp/(idtp+idfp);
    IDR = idtp/(idtp+idfn);
    IDF1 = (2*idtp)/((2*idtp)+idfp+idfn);
    IDRD = idtp/(idtp+idfnD);
    IDF1D = (2*idtp)/((2*idtp)+idfp+idfnD);

end

function [instances, avg_instances, miss_rates, avg_miss, ...
          id_modes, id_mode_counts, recall_precision, avg_recall, ...
          id_recall_precision, avg_id_recall, ...
          num_ids, avg_num_ids, idtp,idfp,idfn,IDP,IDR,IDF1] = validation_statsTogether(enodes)
    val_tab = enodes(1).val_table;
    for cam=2:size(enodes,2)
        val_tab = [val_tab;enodes(cam).val_table];
    end
    
    val_len = size(val_tab,1);
    
    instances = zeros(val_len,1);
    miss_rates = zeros(val_len,1);
    id_modes = zeros(val_len,1);
    id_mode_counts = zeros(val_len,1);
    id_ranked_modes = repmat(ranked_modes(0),val_len,1);
    recall_precision = zeros(val_len,1);
    id_recall_precision = zeros(val_len,1);
    num_ids = zeros(val_len,1);
    successes = 0;
    no_mode = -1;
    idtp = 0;
    idfp = 0;
    idfn = 0;
    IDP = 0;
    IDR = 0;
    IDF1 = 0;
    
    valid_idxs = [];
    for i = 1:val_len
        if nnz(val_tab(i,:)) > 0
            instances(i) = nnz(val_tab(i,:));
            miss_rates(i) = length(find(val_tab(i,:)==-1))/nnz(val_tab(i,:));
            nonzeros = val_tab(i,val_tab(i,:)>0);
            id_ranked_modes(i) = ranked_modes(val_tab(i,:));
            num_ids(i) = length(unique(nonzeros));
            valid_idxs = [valid_idxs; i];
        end
    end
    
    for i = 1:length(valid_idxs)
        idx = valid_idxs(i);
        mode_list = id_ranked_modes(idx).ModeList;
        mode_counts = id_ranked_modes(idx).ModeCounts;
        if ~isempty(mode_list)
            id_modes(idx) = mode_list(1);
            id_mode_counts(idx) = mode_counts(1);
        else
            id_modes(idx) = no_mode;
            id_mode_counts(idx) = 0;
            no_mode = no_mode - 1;
        end
    end
    shared_ids = [];
    unique_ids = unique(id_modes(id_modes>0));

    for i = 1:length(unique_ids)
        sharers = find(id_modes==unique_ids(i));
        if length(sharers) > 1
            shared_ids = [shared_ids; struct('Label',unique_ids(i),'Sharers',sharers)];
        end
    end
    while(~isempty(shared_ids))
        for i = 1:length(shared_ids)
            sharers = shared_ids(i).Sharers;
            starts = zeros(length(sharers),1);
            for j = 1:length(sharers)
                starts(j) = id_ranked_modes(sharers(j)).ModeStarts(1);
            end
            [~,first_start] = min(starts);
            toBeUpdated = find(starts~=first_start);
            for j = 1:length(toBeUpdated)
                id_ranked_modes(sharers(toBeUpdated(j))).ModeList(1) = [];
                id_ranked_modes(sharers(toBeUpdated(j))).ModeCounts(1) = [];
                id_ranked_modes(sharers(toBeUpdated(j))).ModeStarts(1) = [];
            end
        end
        for i = 1:length(valid_idxs)
            idx = valid_idxs(i);
            mode_list = id_ranked_modes(idx).ModeList;
            mode_counts = id_ranked_modes(idx).ModeCounts;
            if ~isempty(mode_list)
                id_modes(idx) = mode_list(1);
                id_mode_counts(idx) = mode_counts(1);
            else
                id_modes(idx) = no_mode;
                id_mode_counts(idx) = 0;
                no_mode = no_mode - 1;
            end
        end
        shared_ids = [];
        unique_ids = unique(id_modes(id_modes>0));

        for i = 1:length(unique_ids)
            sharers = find(id_modes==unique_ids(i));
            if length(sharers) > 1
                shared_ids = [shared_ids; struct('Label',unique_ids(i),'Sharers',sharers)];
            end
        end
    end
    
    for i = 1:val_len
        if nnz(val_tab(i,:)) > 0
            if (id_modes(i) > 0)
                recall_precision(i) = id_mode_counts(i)/nnz(val_tab(i,:));
                id_recall_precision(i) = recall_precision(i)/(1-miss_rates(i));
                successes = successes + length(find(val_tab(i,:)==id_modes(i)));
                idtp = id_mode_counts(i) + idtp;
                idfp = (nnz(val_tab==id_modes(i)) - id_mode_counts(i)) + idfp;
                idfn = (nnz(val_tab(i,:)) - id_mode_counts(i)) + idfn;
                IDP = idtp/(idtp+idfp);
                IDR = idtp/(idtp+idfn);
                IDF1 = (2*idtp)/((2*idtp)+idfp+idfn);
            end
        end
    end
    
    avg_instances = mean(instances(valid_idxs));
    avg_miss = length(find(val_tab==-1))/nnz(val_tab);
    avg_recall = successes/nnz(val_tab);
    avg_id_recall = avg_recall/(1-avg_miss);
    avg_num_ids = mean(num_ids(valid_idxs));
end

function [confusion_matrix] = confusion_analysis(val_table, id_modes)
    num_labels = nnz(id_modes);
    label_idxs = find(id_modes ~= 0);
    confusion_matrix = zeros(num_labels,num_labels+1);
    
    for label1 = 1:num_labels
        num_dets = length(val_table(label1,val_table(label_idxs(label1),:)>0));
        for label2 = 1:num_labels
            confusion_matrix(label1,label2) = ...
                length(val_table(label1,val_table(label_idxs(label1),:)==id_modes(label_idxs(label2)))) ...
                / num_dets;
        end
        confusion_matrix(label1,num_labels+1) = 1 - sum(confusion_matrix(label1,:));
    end

    figure; imshow((1-confusion_matrix));
end

function [confusion_matrix] = sample_confusion_analysis(val_table, id_modes, sample_size)
    num_labels = sample_size;
    label_idxs = find(id_modes ~= 0);
    rand_sample = randperm(length(label_idxs),num_labels);
    label_idxs = label_idxs(rand_sample);
    confusion_matrix = zeros(num_labels,num_labels+1);
    
    for label1 = 1:num_labels
        num_dets = length(val_table(label1,val_table(label_idxs(label1),:)>0));
        for label2 = 1:num_labels
            confusion_matrix(label1,label2) = ...
                length(val_table(label1,val_table(label_idxs(label1),:)==id_modes(label_idxs(label2)))) ...
                / num_dets;
        end
        confusion_matrix(label1,num_labels+1) = 1 - sum(confusion_matrix(label1,:));
    end

    figure; imshow((1-confusion_matrix));
end

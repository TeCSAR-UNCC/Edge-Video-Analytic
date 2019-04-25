classdef EdgeNode
    properties
        detections
        reid_feats
        currFrame
        startFrame
        endFrame
        sourceFrameRate
        frameRate
        
        det_table
    end
    methods
        function obj = EdgeNode(params)
            if nargin == 1
                obj.detections = params.dets;
                obj.reid_feats = params.feats;
                obj.sourceFrameRate = params.sourceFrameRate;
                obj.startFrame = params.startFrame;
                obj.endFrame = params.endFrame;
                obj.currFrame = params.startFrame;
                obj.frameRate = params.outFrameRate;

                pt = personType(0,0,zeros(1,1280),0,0,0,0);
                oh = object_history(0,pt,0,0,0);
                obj.det_table = repmat(oh,params.tab_size,1);
            else
                error('Expecting 1 argument of type camera_params');
            end
        end
        function r = process_step(obj)
            r = obj.currFrame;
        end
        function r = ready(obj, frame)
            if (frame == obj.currFrame)
                r = 1;
            else
                r = 0;
            end
        end
    end
end

function r = personType(currentCamera,label,fv_array,xPos,yPos,height,width)
    r = struct('currentCamera',int32(currentCamera),'label',int32(label), ...
               'fv_array',single(fv_array),'xPos',single(xPos),'yPos', ...
               single(yPos),'height',single(height),'width',single(width));
end

function r = object_history(life,pt,keyCount,reIDFlag,sentToServer)
    r = struct('life',int32(life),'sendObject',pt, ...
               'keyCount',int32(keyCount),'reIDFlag',int32(reIDFlag), ...
               'sentToServer',int32(sentToServer));
end

function r = reIDType(oid,nid)
    r = struct('oldID',oid,'newID',nid);
end

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
    if nargin ~= 7
        currentCamera = 0;
        label = -1;
        fv_array = zeros(1,1280);
        xPos = 0;
        yPos = 0;
        width = 0;
        height = 0;
    end
    r = struct('currentCamera',int32(currentCamera),'label',int32(label), ...
               'fv_array',single(fv_array),'xPos',single(xPos),'yPos', ...
               single(yPos),'height',single(height),'width',single(width));
end
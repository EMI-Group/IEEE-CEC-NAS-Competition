function [CrowdDegree,R] = CalCrowdDegree(PopObj,N,k)

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Normalization
    fmax  = max(PopObj,[],1);
    fmin  = min(PopObj,[],1);
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    
    %% Calculate the radius of niche (The average distance to the k nearest individuals)
    distance = pdist2(PopObj,PopObj);
    distance(logical(eye(length(distance)))) = inf;
    sd = sort(distance,2);
    r  = mean(sd(:,min(k,size(sd,2))));
    
    %% Calculate the crowd degree
    R  = min(distance./r,1);
	CrowdDegree = 1 - prod(R,2);
end
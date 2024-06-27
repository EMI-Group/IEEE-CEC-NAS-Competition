function [Ie,I,C] = CalculateIndicator(PopuObj,N)

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

%% Calculate the fitness of each solution
    Obj = PopuObj;
    Obj = (Obj-repmat(min(Obj),N,1))./(repmat(max(Obj)-min(Obj),N,1));
    I = zeros(N);
    for i = 1 : N
        for j = 1 : N
            I(i,j) = max(Obj(i,:)-Obj(j,:));
        end
    end
    C = max(abs(I));
    Ie = sum(-exp(-I./repmat(C,N,1)/0.05)) + 1;
end
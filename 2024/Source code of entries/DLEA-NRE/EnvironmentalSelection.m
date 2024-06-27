function [Population,FrontNo] = EnvironmentalSelection(Population,Problem,alpha,k,Eva_Obj)
% The environmental selection of DLEA-NRE

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Eva_Obj,Population.cons,Problem.N);
    Next = FrontNo < MaxFNo;
    
	%% Choose N-sum(Next) individuals from MaxFNo by using two stragety
	%Set parameters
	final_choose = Problem.N-sum(Next);
	
	Convergence_number = max(ceil(final_choose*alpha*(1-Problem.FE/Problem.maxFE)),1);
	Diversity_number = final_choose - Convergence_number;
	
	%% Calculate the indicator Ie of every solution
	[Ie,~,~] = CalculateIndicator(Eva_Obj,length(Eva_Obj));
	
	%% Select the convergence solutions in the last front based on their Ie indicator
	Last = find(FrontNo==MaxFNo);
	[~,Rank] = sort(Ie(Last),'descend');
	Next(Last(Rank(1:Convergence_number))) = true;
    
    

	%% Select the diversity solutions in the last front based on their crowd degree
    unSelected = Last(Next(Last)~=true);
    siz = length(unSelected);
    if(siz > 0)
        [~,R] = CalCrowdDegree(Eva_Obj(unSelected,:),length(unSelected),k);
        for i = 1 : siz - Diversity_number
            [~,worst] = max(1 - prod(R,2));
            R(worst,:) = [];
            R(:,worst) = [];
            unSelected(worst) = [];
        end
        Next(unSelected) = true;
    end
    
    %% Find extreme points to maintain diversity(Priority is given to maintaining extreme points)
    [~, Extreme] = FindExtremeSolution(Eva_Obj);
    % Check whether the extremum is selected, if not, keep and delete the most recent selected solution
    for i  = 1 : size(Extreme,2)
        if Next(Extreme(i)) == 0
            % calculate distance from extreme i to all points in next
            choose = find(Next == 1);
            distance = pdist2(Eva_Obj(Extreme(i),:),Eva_Obj(choose,:));
            [~,minInd] = min(distance);
            deleteInd = choose(minInd);
            Next(Extreme(i)) = 1;
            Next(deleteInd) = 0;  
        end
    end
    
     %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
end
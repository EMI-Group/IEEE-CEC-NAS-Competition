function [Population,FrontNo] = EnvironmentalSelection(Population,Problem,alpha,k)
% The environmental selection of DLEA-Niche

%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,Problem.N);
    Next = FrontNo < MaxFNo;
    
	%% Choose N-sum(Next) individuals from MaxFNo by using two strategy
	%Set parameters
	final_choose = Problem.N-sum(Next);
	
	Convergence_number = max(ceil(final_choose*alpha*(1-Problem.FE/Problem.maxFE)),1);
	Diversity_number = final_choose - Convergence_number;
	
	%%Calculate the indicator Ie of every solution
	[Ie,~,~] = CalculateIndicator(Population.objs,length(Population.objs));
	
	%% Select the convergence solutions in the last front based on their Ie indicator
	Last = find(FrontNo==MaxFNo);
	[~,Rank] = sort(Ie(Last),'descend');
	Next(Last(Rank(1:Convergence_number))) = true;
    	
	%% Select the diversity solutions in the last front based on their crowd degree
    unSelected = Last(Next(Last)~=true);
    siz = length(unSelected);
    if(siz > 0)
        [~,R] = CalCrowdDegree(Population(unSelected).objs,length(unSelected),k);
        for i = 1 : siz - Diversity_number
            [~,worst] = max(1 - prod(R,2));
            R(worst,:) = [];
            R(:,worst) = [];
            unSelected(worst) = [];
        end
        Next(unSelected) = true;
    end
    
     %% Population for next generation
    Population = Population(Next);
    FrontNo    = FrontNo(Next);
end
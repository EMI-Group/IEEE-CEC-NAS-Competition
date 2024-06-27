classdef DLEANRE < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <constrained/none>
% Dynamic learning evolution algorithm with niche diversity maintain
% strategy and robust objective strategy
% alpha --- 0.9 --- Convergence factor
% k     --- 3 --- Niche paremeter

%------------------------------- Reference --------------------------------
% G Li, GG Wang, J Dong, WC Yeh, K Li, DLEA: A dynamic learning evolution 
% algorithm for many-objective optimization, Information sciences, 2021, 
% 574, 567-589
%--------------------------------------------------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".m
%--------------------------------------------------------------------------


	methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha,k,sampling] = Algorithm.ParameterSet(0.9,3,5);
            
            %% Generate random population
            Population = Problem.Initialization();
            
            %% Obtain multiple evaluation samples
            [samples,maxIter] = SampleForTwiceEvaluation(Problem.maxFE,Problem.N,sampling);

            %% Optimization
            while Algorithm.NotTerminated(Population)
                Iter = Problem.FE / Problem.N;
                if ismember(Iter,samples)
                    Old_Obj = Population.objs;
                    % re-Evaluation
                    Population = Problem.Evaluation(Population.decs);
                    New_Obj = Population.objs;
                    Eva_Obj = [];
                    for i = 1 : Problem.M
                        Eva_Obj = [Eva_Obj,min([Old_Obj(:,i),New_Obj(:,i)],[],2)];
                    end
                else
                    Eva_Obj = Population.objs;
                end
                
				MatingPool = TournamentSelection(2,Problem.N,-CalculateIndicator(Eva_Obj,length(Eva_Obj)));
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,~] = EnvironmentalSelection([Population,Offspring],Problem,alpha,k,[Eva_Obj;Offspring.objs]);                
            end
        end
    end
end


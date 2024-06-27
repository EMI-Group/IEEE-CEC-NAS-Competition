classdef GrSMEA_NCHU < ALGORITHM
% <multi/many> <real/integer>
% Grid-Based Evolutionary Algorithm Assisted by Self-Organizing Map for Multiobjective Neural Architecture Search in Real-Time Semantic Segmentation
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            Population1    =Problem.Initialization();
            N= Problem.N;
            Div = [0 45 15 10 9 9 8 8 10 12];
            div = Algorithm.ParameterSet(Div(min(Problem.M,10)));
            [D,tau0,H] = Algorithm.ParameterSet(repmat(ceil(Problem.N.^(1/(Problem.M-1))),1,Problem.M-1),0.7,5);
            Problem.N  = prod(D);
            sigma0     = sqrt(sum(D.^2)/(Problem.M-1))/2;
            %% Generate random population
            Population2 = Problem.Initialization();
            
            %% Initialize the SOM
            S = Population2.decs;
            % Weight vector of each neuron
            W = S;
            [LDiS,B] = Initialize_SOM(S,D,H);
            
            %% Optimization
            while Algorithm.NotTerminated(Population1)
                W = UpdateSOM(S,W,Problem.FE,Problem.maxFE,LDiS,sigma0,tau0);
                MatingPool1 = MatingSelection1(Population1.objs,div);
                Offspring1  = OperatorGA(Problem,Population1(MatingPool1));
                XU = Associate(Population2,W,Problem.N); 
                MatingPool2 = MatingSelection2(XU,Problem.N,B);
                A         = Population2.decs;
                rand_=randsample(Problem.N,N);
                Offspring2 = OperatorGA(Problem,[Population2(XU(rand_)),Population2(MatingPool2(rand_))]);
                Population1 = EnvironmentalSelection1([Population1,Offspring1,Offspring2],N,div);
                Population2 = UpdateCA(Population2,[Offspring1,Offspring2],Problem.N);
                S = setdiff(Population2.decs,A,'rows');  
            end
        end
    end
end
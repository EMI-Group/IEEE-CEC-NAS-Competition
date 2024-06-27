classdef MOEA_AP < ALGORITHM
    % <multi> <real/integer> <constrained/none>
    % MOEA-AP
    
    %------------------------------- Reference --------------------------------
    % J. Shen and J. Liu, MOEA-AP for Multiobjective Neural Architecture Search 
    % Challenge for Real-Time Semantic Segmentation in IEEE WCCI 2024
    %--------------------------------------------------------------------------
    % Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
    % research purposes. All publications which use this platform or any code
    % in the platform should acknowledge the use of "PlatEMO" and reference "Ye
    % Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
    % for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
    % Computational Intelligence Magazine, 2017, 12(4): 73-87".
    %--------------------------------------------------------------------------
    
    % The code could be used in PlatEMO v4.5 !!!
    methods
        function main(Algorithm,Problem)
            %% Generate random population           
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = AGE_EnvironmentalSelection(Population,Problem.N);
            B  = eye(Problem.D);
            m  = 0.5*(Problem.upper - Problem.lower);
            ps = 0.5;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = ARSBX(Problem,Population(MatingPool),{B,m,ps});
                [Population,FrontNo,CrowdDis] = AGE_EnvironmentalSelection([Population,Offspring],Problem.N);
                [B,m,ps,Population] = UpdateParameter(Problem,Population);
            end
        end
    end
end
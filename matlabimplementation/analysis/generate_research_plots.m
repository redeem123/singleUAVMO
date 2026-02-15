%% Research Analysis: Performance Plotting
% This script generates Pareto Fronts and Convergence Curves for all algorithms.

clear; clc;
% Initialize paths
scriptDir = fileparts(mfilename('fullpath'));
run(fullfile(scriptDir, '..', 'startup.m'));

resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');

% Identify Algorithm Folders
algoFolders = dir(resultsDir);
algoFolders = algoFolders([algoFolders.isdir]);
algoFolders = algoFolders(~strncmp({algoFolders.name}, '.', 1) & ~strcmp({algoFolders.name}, 'Plots'));

if isempty(algoFolders)
    error('No results found. Run scripts/run_benchmark.m first.');
end

% Create directory for plots
plotDir = fullfile(resultsDir, 'Plots');
if ~isfolder(plotDir), mkdir(plotDir); end

for a = 1:numel(algoFolders)
    algoName = algoFolders(a).name;
    algoDir = fullfile(resultsDir, algoName);
    
    fprintf('Processing Algorithm: %s\n', algoName);
    
    % Identify Problem Folders for this Algorithm
    probFolders = dir(algoDir);
    probFolders = probFolders([probFolders.isdir]);
    probFolders = probFolders(~strncmp({probFolders.name}, '.', 1));
    
    for i = 1:numel(probFolders)
        problemName = probFolders(i).name;
        fprintf('  - Plotting results for: %s\n', problemName);
        
        currentProbDir = fullfile(algoDir, problemName);
        
        %% 1. Pareto Front (Final Generation of Run 1)
        run1Dir = fullfile(currentProbDir, 'Run_1');
        d = dir(fullfile(run1Dir, 'bp_*.mat'));
        
        if ~isempty(d)
            objs = [];
            for j = 1:numel(d)
                tmp = load(fullfile(run1Dir, d(j).name));
                if isfield(tmp, 'dt_sv') && isfield(tmp.dt_sv, 'objs')
                    objs = [objs; tmp.dt_sv.objs(:)'];
                end
            end

            objs = objs(all(isfinite(objs), 2), :);

            if ~isempty(objs)
                FrontNo = NDSort(objs, 1);
                front = objs(FrontNo == 1, :);

                fig1 = figure('Visible', 'off');
                parallelcoords(front, 'LineWidth', 0.8);
                grid on;
                title(['Pareto Front (Parallel Coordinates) (' algoName '): ' strrep(problemName, '_', ' ')]);
                set(gca, 'XTickLabel', {'J1 Path', 'J2 Threat', 'J3 Altitude', 'J4 Smooth'});
                set(fig1, 'Visible', 'on');
                saveas(fig1, fullfile(plotDir, ['Pareto_' algoName '_' problemName '.fig']));
                close(fig1);
            end
        end
        
        %% 2. Convergence Curve (HV over Generations)
        hv_history = [];
        for r = 1:5 
            hvFile = fullfile(currentProbDir, sprintf('Run_%d', r), 'gen_hv.mat');
            if exist(hvFile, 'file')
                tmp = load(hvFile);
                hv_history = [hv_history, tmp.gen_hv(:,1)];
            end
        end
        
        if ~isempty(hv_history)
            fig2 = figure('Visible', 'off');
            mean_hv = mean(hv_history, 2);
            std_hv = std(hv_history, 0, 2);
            gens = 1:length(mean_hv);
            
            fill([gens, fliplr(gens)], [mean_hv-std_hv; flipud(mean_hv+std_hv)], ...
                [0.8 0.8 1], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
            hold on;
            plot(gens, mean_hv, 'b-', 'LineWidth', 2);
            grid on;
            xlabel('Generation');
            ylabel('Hypervolume (HV)');
            title(['Convergence (' algoName '): ' strrep(problemName, '_', ' ')]);
            legend('Std Dev', 'Mean HV', 'Location', 'southeast');
            set(fig2, 'Visible', 'on');
            saveas(fig2, fullfile(plotDir, ['Convergence_' algoName '_' problemName '.fig']));
            close(fig2);
        end
        
        %% 3. Representative 3D Path Visualization
        terrainFile = fullfile(fileparts(mfilename('fullpath')), '..', 'problems', sprintf('terrainStruct_%s.mat', problemName));
        if exist(terrainFile, 'file') && ~isempty(d)
            data = load(terrainFile);
            sid = randi([1, length(d)], 1);
            tmp_sol = load(fullfile(run1Dir, d(sid).name));
            sol = tmp_sol.dt_sv.path;
            
            terrain = data.terrainStruct.H;
            X = data.terrainStruct.X;
            Y = data.terrainStruct.Y;

            fig3 = figure('Visible', 'off');
            surf(X, Y, terrain, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
            colormap(parula);
            hold on;
            plot3(sol(:,1), sol(:,2), sol(:,3), 'r-', 'LineWidth', 2.5);
            scatter3(sol(1,1), sol(1,2), sol(1,3), 80, 'g', 'filled');
            scatter3(sol(end,1), sol(end,2), sol(end,3), 80, 'm', 'filled');
            
            xlabel('X'); ylabel('Y'); zlabel('Altitude');
            title(['3D Path (' algoName '): ' strrep(problemName, '_', ' ')]);
            view(45, 30);
            grid on;
            set(fig3, 'Visible', 'on');
            saveas(fig3, fullfile(plotDir, ['Path3D_' algoName '_' problemName '.fig']));
            close(fig3);
        end
    end
end

fprintf('All plots saved to: %s\n', plotDir);

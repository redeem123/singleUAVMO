function generate_research_plots_cli(projectRoot, resultsDir)
%GENERATE_RESEARCH_PLOTS_CLI Generate benchmark plots from results folder.
%   This CLI-oriented variant is called from Python and writes PNG plots to:
%   <resultsDir>/Plots

    if nargin < 1 || isempty(projectRoot)
        projectRoot = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || isempty(resultsDir)
        resultsDir = fullfile(projectRoot, 'results');
    end

    if ~isfolder(resultsDir)
        error('Results directory not found: %s', resultsDir);
    end

    plotDir = fullfile(resultsDir, 'Plots');
    if ~isfolder(plotDir)
        mkdir(plotDir);
    end

    algoFolders = dir(resultsDir);
    algoFolders = algoFolders([algoFolders.isdir]);
    algoFolders = algoFolders(~strncmp({algoFolders.name}, '.', 1) & ~strcmp({algoFolders.name}, 'Plots'));
    if isempty(algoFolders)
        error('No algorithm results found in %s', resultsDir);
    end

    for a = 1:numel(algoFolders)
        algoName = algoFolders(a).name;
        algoDir = fullfile(resultsDir, algoName);
        fprintf('Processing algorithm: %s\n', algoName);

        probFolders = dir(algoDir);
        probFolders = probFolders([probFolders.isdir]);
        probFolders = probFolders(~strncmp({probFolders.name}, '.', 1));

        for p = 1:numel(probFolders)
            problemName = probFolders(p).name;
            fprintf('  - Problem: %s\n', problemName);
            problemDir = fullfile(algoDir, problemName);

            runDirs = dir(fullfile(problemDir, 'Run_*'));
            runDirs = runDirs([runDirs.isdir]);
            if isempty(runDirs)
                continue;
            end

            runOneDir = fullfile(problemDir, 'Run_1');
            if ~isfolder(runOneDir)
                runOneDir = fullfile(problemDir, runDirs(1).name);
            end

            %% 1) Pareto parallel-trend plot from Run_1
            bpFiles = dir(fullfile(runOneDir, 'bp_*.mat'));
            if ~isempty(bpFiles)
                objs = [];
                for i = 1:numel(bpFiles)
                    data = load(fullfile(runOneDir, bpFiles(i).name));
                    if isfield(data, 'dt_sv') && isfield(data.dt_sv, 'objs')
                        v = data.dt_sv.objs(:)';
                        if numel(v) >= 4
                            objs = [objs; v(1:4)]; %#ok<AGROW>
                        end
                    end
                end

                if ~isempty(objs)
                    objs = objs(all(isfinite(objs), 2), :);
                    if ~isempty(objs)
                        frontNo = NDSort(objs, 1);
                        front = objs(frontNo == 1, :);
                        if ~isempty(front)
                            fig = figure('Visible', 'off');
                            ax = axes(fig); %#ok<LAXES>
                            hold(ax, 'on');
                            for r = 1:size(front, 1)
                                plot(ax, 1:4, front(r, :), '-', 'Color', [0.8, 0.1, 0.1], 'LineWidth', 0.8);
                            end
                            hold(ax, 'off');
                            grid(ax, 'on');
                            title(ax, sprintf('Pareto Front: %s / %s', algoName, problemName), 'Interpreter', 'none');
                            xlim(ax, [1 4]);
                            xticks(ax, 1:4);
                            xticklabels(ax, {'J1 Path', 'J2 Threat', 'J3 Altitude', 'J4 Smooth'});
                            ylabel(ax, 'Objective Value');
                            exportgraphics(fig, fullfile(plotDir, sprintf('Pareto_%s_%s.png', algoName, problemName)), 'Resolution', 220);
                            close(fig);
                        end
                    end
                end
            end

            %% 2) HV convergence from first 5 runs (if available)
            hvSeries = [];
            maxRuns = min(5, numel(runDirs));
            for r = 1:maxRuns
                hvFile = fullfile(problemDir, sprintf('Run_%d', r), 'gen_hv.mat');
                if ~exist(hvFile, 'file')
                    continue;
                end
                hvData = load(hvFile);
                if isfield(hvData, 'gen_hv') && size(hvData.gen_hv, 2) >= 1
                    hvSeries = [hvSeries, hvData.gen_hv(:, 1)]; %#ok<AGROW>
                end
            end
            if ~isempty(hvSeries)
                meanHv = mean(hvSeries, 2, 'omitnan');
                stdHv = std(hvSeries, 0, 2, 'omitnan');
                gen = (1:numel(meanHv))';
                fig = figure('Visible', 'off');
                ax = axes(fig); %#ok<LAXES>
                fill(ax, [gen; flipud(gen)], [meanHv - stdHv; flipud(meanHv + stdHv)], ...
                    [0.8 0.85 1.0], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
                hold(ax, 'on');
                plot(ax, gen, meanHv, 'b-', 'LineWidth', 2);
                hold(ax, 'off');
                grid(ax, 'on');
                xlabel(ax, 'Generation');
                ylabel(ax, 'Hypervolume (HV)');
                title(ax, sprintf('Convergence: %s / %s', algoName, problemName), 'Interpreter', 'none');
                legend(ax, {'Std Dev', 'Mean HV'}, 'Location', 'southeast');
                exportgraphics(fig, fullfile(plotDir, sprintf('Convergence_%s_%s.png', algoName, problemName)), 'Resolution', 220);
                close(fig);
            end

            %% 3) 3D path plot (prefer feasible path from any run)
            terrainFile = fullfile(projectRoot, 'problems', sprintf('terrainStruct_%s.mat', problemName));
            if exist(terrainFile, 'file')
                terrainData = load(terrainFile);
                if isfield(terrainData, 'terrainStruct')
                    [pathFound, pathXYZ] = pick_path_for_plot(problemDir, runDirs);
                    if pathFound
                        H = terrainData.terrainStruct.H;
                        X = terrainData.terrainStruct.X;
                        Y = terrainData.terrainStruct.Y;

                        fig = figure('Visible', 'off');
                        surf(X, Y, H, 'EdgeColor', 'none', 'FaceAlpha', 1.0);
                        colormap(parula);
                        hold on;
                        plot3(pathXYZ(:, 1), pathXYZ(:, 2), pathXYZ(:, 3), 'r-', 'LineWidth', 2.5);
                        scatter3(pathXYZ(1, 1), pathXYZ(1, 2), pathXYZ(1, 3), 80, 'g', 'filled');
                        scatter3(pathXYZ(end, 1), pathXYZ(end, 2), pathXYZ(end, 3), 80, 'm', 'filled');
                        xlabel('X'); ylabel('Y'); zlabel('Altitude');
                        title(sprintf('3D Path: %s / %s', algoName, problemName), 'Interpreter', 'none');
                        view(45, 30);
                        grid on;
                        hold off;
                        exportgraphics(fig, fullfile(plotDir, sprintf('Path3D_%s_%s.png', algoName, problemName)), 'Resolution', 220);
                        close(fig);
                    end
                end
            end
        end
    end

    fprintf('MATLAB plots generated at: %s\n', plotDir);
end

function [ok, pathXYZ] = pick_path_for_plot(problemDir, runDirs)
    ok = false;
    pathXYZ = [];

    % First pass: find a feasible path (all finite objectives) across runs.
    for r = 1:numel(runDirs)
        runPath = fullfile(problemDir, runDirs(r).name);
        popFile = fullfile(runPath, 'final_popobj.mat');
        if ~exist(popFile, 'file')
            continue;
        end
        popData = load(popFile);
        if ~isfield(popData, 'PopObj')
            continue;
        end
        popObj = popData.PopObj;
        if ndims(popObj) ~= 2
            continue;
        end
        if size(popObj, 2) ~= 4 && size(popObj, 1) == 4
            popObj = popObj';
        end
        finiteRows = all(isfinite(popObj), 2);
        idx = find(finiteRows);
        if isempty(idx)
            continue;
        end
        candidate = idx(randi(numel(idx)));
        bpFile = fullfile(runPath, sprintf('bp_%d.mat', candidate));
        if exist(bpFile, 'file')
            data = load(bpFile);
            if isfield(data, 'dt_sv') && isfield(data.dt_sv, 'path')
                p = data.dt_sv.path;
                if isnumeric(p) && size(p, 2) == 3 && size(p, 1) >= 2
                    pathXYZ = p;
                    ok = true;
                    return;
                end
            end
        end
    end

    % Fallback: any available path from Run_1.
    runOne = fullfile(problemDir, 'Run_1');
    if ~isfolder(runOne) && ~isempty(runDirs)
        runOne = fullfile(problemDir, runDirs(1).name);
    end
    bpFiles = dir(fullfile(runOne, 'bp_*.mat'));
    if isempty(bpFiles)
        return;
    end
    sid = randi(numel(bpFiles));
    data = load(fullfile(runOne, bpFiles(sid).name));
    if isfield(data, 'dt_sv') && isfield(data.dt_sv, 'path')
        p = data.dt_sv.path;
        if isnumeric(p) && size(p, 2) == 3 && size(p, 1) >= 2
            pathXYZ = p;
            ok = true;
        end
    end
end

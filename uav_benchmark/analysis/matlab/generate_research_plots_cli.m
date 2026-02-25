function generate_research_plots_cli(projectRoot, resultsDir)
%GENERATE_RESEARCH_PLOTS_CLI Generate benchmark plots from results folder.
%   Writes FIG plots to: <resultsDir>/Plots

    if nargin < 1 || isempty(projectRoot)
        projectRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
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
    algoFolders = filter_algorithm_dirs(resultsDir, algoFolders);
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

            %% 1) Pareto parallel-trend plot from final_popobj.mat
            popFile = fullfile(runOneDir, 'final_popobj.mat');
            if exist(popFile, 'file')
                popData = load(popFile);
                if isfield(popData, 'PopObj')
                    objs = double(popData.PopObj);
                    if size(objs, 2) ~= 4 && size(objs, 1) == 4
                        objs = objs';
                    end
                    if size(objs, 2) >= 4
                        objs = objs(:, 1:4);
                        objs = objs(all(isfinite(objs), 2), :);
                        if ~isempty(objs)
                            frontMask = non_dominated_mask(objs);
                            front = objs(frontMask, :);
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
                                save_plot_fig(fig, fullfile(plotDir, sprintf('Pareto_%s_%s.fig', algoName, problemName)));
                                close(fig);
                            end
                        end
                    end
                end
            end

            %% 2) 3D path plot (prefer fleet_paths, fallback to feasible bp path)
            terrainProblem = regexprep(problemName, '_uav\d+$', '');
            terrainFile = fullfile(projectRoot, 'problems', sprintf('terrainStruct_%s.mat', terrainProblem));
            if exist(terrainFile, 'file')
                terrainData = load(terrainFile);
                if isfield(terrainData, 'terrainStruct')
                    [pathFound, pathSet, pathNames] = pick_paths_for_plot(problemDir, runDirs);
                    if pathFound
                        H = double(terrainData.terrainStruct.H);
                        X = double(terrainData.terrainStruct.X);
                        Y = double(terrainData.terrainStruct.Y);
                        if isvector(X) && isvector(Y)
                            [XX, YY] = meshgrid(X, Y);
                        else
                            XX = X;
                            YY = Y;
                        end

                        fig = figure('Visible', 'off');
                        surf(XX, YY, H, 'EdgeColor', 'none', 'FaceAlpha', 1.0);
                        colormap(parula);
                        hold on;
                        cmap = lines(numel(pathSet));
                        for k = 1:numel(pathSet)
                            pathXYZ = pathSet{k};
                            c = cmap(k, :);
                            plot3(pathXYZ(:, 1), pathXYZ(:, 2), pathXYZ(:, 3), '-', 'Color', c, 'LineWidth', 2.2);
                            scatter3(pathXYZ(1, 1), pathXYZ(1, 2), pathXYZ(1, 3), 42, c, 'filled');
                            scatter3(pathXYZ(end, 1), pathXYZ(end, 2), pathXYZ(end, 3), 52, c, 'd', 'filled');
                        end
                        xlabel('X'); ylabel('Y'); zlabel('Altitude');
                        title(sprintf('3D Path: %s / %s', algoName, problemName), 'Interpreter', 'none');
                        view(45, 30);
                        grid on;
                        if numel(pathNames) > 1
                            legend(pathNames, 'Interpreter', 'none', 'Location', 'bestoutside');
                        end
                        hold off;
                        save_plot_fig(fig, fullfile(plotDir, sprintf('Path3D_%s_%s.fig', algoName, problemName)));
                        close(fig);
                    end
                end
            end
        end
    end

    fprintf('MATLAB plots generated at: %s\n', plotDir);
end

function [ok, pathSet, pathNames] = pick_paths_for_plot(problemDir, runDirs)
    ok = false;
    pathSet = {};
    pathNames = {};

    % First pass: prefer fleet_paths.mat and collect all UAV paths in the run.
    for r = 1:numel(runDirs)
        runPath = fullfile(problemDir, runDirs(r).name);
        fleetFile = fullfile(runPath, 'fleet_paths.mat');
        if exist(fleetFile, 'file')
            data = load(fleetFile);
            fns = fieldnames(data);
            for i = 1:numel(fns)
                p = data.(fns{i});
                if isnumeric(p) && size(p, 2) == 3 && size(p, 1) >= 2 && all(isfinite(p(:)))
                    pathSet{end+1} = p; %#ok<AGROW>
                    pathNames{end+1} = fns{i}; %#ok<AGROW>
                end
            end
            if ~isempty(pathSet)
                ok = true;
                return;
            end
        end
    end

    % Fallback: choose a feasible single path from Run_1 bp files.
    runOne = fullfile(problemDir, 'Run_1');
    if ~isfolder(runOne) && ~isempty(runDirs)
        runOne = fullfile(problemDir, runDirs(1).name);
    end
    bpFiles = dir(fullfile(runOne, 'bp_*.mat'));
    if isempty(bpFiles)
        return;
    end

    [~, order] = sort(extract_bp_index({bpFiles.name}));
    bpFiles = bpFiles(order);
    bpIdx = extract_bp_index({bpFiles.name});
    countHint = max([bpIdx(:); numel(bpFiles)]);
    feasibleIdx = single_uav_feasible_indices(runOne, countHint);
    if ~isempty(feasibleIdx)
        keep = ismember(bpIdx, feasibleIdx);
        if any(keep)
            bpFiles = bpFiles(keep);
            bpIdx = bpIdx(keep);
        end
    end

    % Pick the lowest-index feasible path to keep plots deterministic.
    [~, bestPos] = min(bpIdx);
    data = load(fullfile(runOne, bpFiles(bestPos).name));
    if isfield(data, 'dt_sv') && isfield(data.dt_sv, 'path')
        p = data.dt_sv.path;
        if isnumeric(p) && size(p, 2) == 3 && size(p, 1) >= 2
            pathSet{1} = p;
            pathNames{1} = sprintf('bp_%d', bpIdx(bestPos));
            ok = true;
        end
    end
end

function idx = extract_bp_index(names)
    idx = zeros(size(names));
    for i = 1:numel(names)
        token = regexp(names{i}, '^bp_(\d+)\.mat$', 'tokens', 'once');
        if isempty(token)
            idx(i) = i;
        else
            idx(i) = str2double(token{1});
        end
    end
end

function idx = single_uav_feasible_indices(runDir, countHint)
    idx = [];
    if countHint <= 0
        return;
    end
    mask = true(1, countHint);
    applied = false;

    missionFile = fullfile(runDir, 'mission_stats.mat');
    if exist(missionFile, 'file')
        ms = load(missionFile);
        if isfield(ms, 'feasible')
            m = full_length_bool_mask(ms.feasible, countHint, 0.5);
            if ~isempty(m)
                mask = mask & m;
                applied = true;
            end
        end
        keys = {'separationViolation', 'collisionViolation'};
        for k = 1:numel(keys)
            key = keys{k};
            if isfield(ms, key)
                v = full_length_bool_mask(ms.(key), countHint, 0.5);
                if ~isempty(v)
                    mask = mask & ~v;
                    applied = true;
                end
            end
        end
    end

    if applied
        idx = find(mask);
        return;
    end

    popFile = fullfile(runDir, 'final_popobj.mat');
    if ~exist(popFile, 'file')
        return;
    end
    popData = load(popFile);
    if ~isfield(popData, 'PopObj')
        return;
    end
    popObj = popData.PopObj;
    if size(popObj, 2) ~= 4 && size(popObj, 1) == 4
        popObj = popObj';
    end
    if size(popObj, 1) ~= countHint
        return;
    end
    idx = find(all(isfinite(popObj), 2))';
end

function mask = full_length_bool_mask(values, countHint, threshold)
    flat = values(:)';
    if numel(flat) ~= countHint
        mask = [];
        return;
    end
    mask = flat > threshold;
end

function keep = non_dominated_mask(points)
    n = size(points, 1);
    keep = true(n, 1);
    for i = 1:n
        if ~keep(i)
            continue;
        end
        for j = 1:n
            if i == j || ~keep(j)
                continue;
            end
            if all(points(j, :) <= points(i, :)) && any(points(j, :) < points(i, :))
                keep(i) = false;
                break;
            end
        end
    end
end

function out = filter_algorithm_dirs(resultsDir, in)
    keep = false(size(in));
    for i = 1:numel(in)
        algoDir = fullfile(resultsDir, in(i).name);
        probs = dir(algoDir);
        probs = probs([probs.isdir]);
        probs = probs(~strncmp({probs.name}, '.', 1));
        for p = 1:numel(probs)
            if ~isempty(dir(fullfile(algoDir, probs(p).name, 'Run_*')))
                keep(i) = true;
                break;
            end
        end
    end
    out = in(keep);
end

function save_plot_fig(fig, outFile)
    try
        savefig(fig, outFile);
    catch
        saveas(fig, outFile);
    end
end

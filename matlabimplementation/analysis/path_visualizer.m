function outPath = path_visualizer(problemName, runNum, varargin)
%PATH_VISUALIZER Interactive 3D visualization for one run/path.
%   outPath = path_visualizer(problemName, runNum, Name, Value, ...)
%
% Name-Value options:
%   'algorithm'    : algorithm folder name under results (default: 'NMOPSO')
%   'projectRoot'  : pythonimplementation root (default: auto-detect)
%   'resultsDir'   : results root directory (default: <projectRoot>/results)
%   'pathIndex'    : choose specific bp_<index>.mat (default: [])
%   'feasibleOnly' : choose from finite PopObj rows only (default: true)
%   'savePng'      : save PNG under results/Plots/Paths (default: true)

    parser = inputParser;
    addParameter(parser, 'algorithm', 'NMOPSO');
    addParameter(parser, 'projectRoot', '');
    addParameter(parser, 'resultsDir', '');
    addParameter(parser, 'pathIndex', []);
    addParameter(parser, 'feasibleOnly', true);
    addParameter(parser, 'savePng', true);
    parse(parser, varargin{:});
    opts = parser.Results;

    scriptPath = fileparts(mfilename('fullpath'));
    defaultProjectRoot = fileparts(fileparts(scriptPath));
    if isempty(opts.projectRoot)
        projectRoot = defaultProjectRoot;
    else
        projectRoot = char(opts.projectRoot);
    end
    if isempty(opts.resultsDir)
        resultsDir = fullfile(projectRoot, 'results');
    else
        resultsDir = char(opts.resultsDir);
    end

    terrainFile = fullfile(projectRoot, 'problems', sprintf('terrainStruct_%s.mat', problemName));
    if ~exist(terrainFile, 'file')
        error('Terrain file not found: %s', terrainFile);
    end
    terrainData = load(terrainFile);
    if ~isfield(terrainData, 'terrainStruct')
        error('terrainStruct missing in %s', terrainFile);
    end

    runDir = fullfile(resultsDir, opts.algorithm, problemName, sprintf('Run_%d', runNum));
    if ~isfolder(runDir)
        legacyRunDir = fullfile(resultsDir, problemName, sprintf('Run_%d', runNum));
        if isfolder(legacyRunDir)
            runDir = legacyRunDir;
        else
            error('Run directory not found: %s', runDir);
        end
    end

    bpInfo = collect_bp_info(runDir);
    if isempty(bpInfo)
        error('No bp_*.mat files found in %s', runDir);
    end

    selected = [];
    if ~isempty(opts.pathIndex)
        requested = double(opts.pathIndex);
        k = find([bpInfo.idx] == requested, 1, 'first');
        if isempty(k)
            error('bp_%d.mat not found in %s', requested, runDir);
        end
        selected = bpInfo(k);
    else
        candidates = bpInfo;
        if logical(opts.feasibleOnly)
            feasibleIdx = feasible_indices(runDir);
            if ~isempty(feasibleIdx)
                keep = ismember([bpInfo.idx], feasibleIdx);
                candidates = bpInfo(keep);
            end
        end
        if isempty(candidates)
            error('No candidate paths available in %s with current filters.', runDir);
        end
        selected = candidates(randi(numel(candidates)));
    end

    solutionData = load(selected.filepath);
    if ~isfield(solutionData, 'dt_sv') || ~isfield(solutionData.dt_sv, 'path')
        error('Invalid bp file format: %s', selected.filepath);
    end
    path = solutionData.dt_sv.path;
    if ~isnumeric(path) || size(path, 2) ~= 3 || size(path, 1) < 2
        error('Invalid path matrix in: %s', selected.filepath);
    end

    H = terrainData.terrainStruct.H;
    X = terrainData.terrainStruct.X;
    Y = terrainData.terrainStruct.Y;

    fig = figure('Name', sprintf('3D Path: %s / %s / Run %d / bp_%d', ...
        opts.algorithm, problemName, runNum, selected.idx), ...
        'Color', 'w');
    surf(X, Y, H, 'EdgeColor', 'none', 'FaceAlpha', 1.0);
    colormap(parula);
    hold on;
    plot3(path(:, 1), path(:, 2), path(:, 3), 'r-', 'LineWidth', 2.5);
    scatter3(path(1, 1), path(1, 2), path(1, 3), 80, 'g', 'filled');
    scatter3(path(end, 1), path(end, 2), path(end, 3), 80, 'm', 'filled');
    xlabel('X');
    ylabel('Y');
    zlabel('Altitude');
    title(sprintf('3D Path: %s / %s / Run %d / bp_%d', ...
        opts.algorithm, strrep(problemName, '_', ' '), runNum, selected.idx), ...
        'Interpreter', 'none');
    legend({'Terrain', 'UAV Path', 'Start', 'End'}, 'Location', 'best');
    view(45, 30);
    grid on;
    hold off;

    outPath = '';
    if logical(opts.savePng)
        plotDir = fullfile(resultsDir, 'Plots', 'Paths');
        if ~isfolder(plotDir)
            mkdir(plotDir);
        end
        outPath = fullfile(plotDir, sprintf('Path_%s_%s_run%d_bp%d.png', ...
            opts.algorithm, problemName, runNum, selected.idx));
        exportgraphics(fig, outPath, 'Resolution', 220);
    end
end

function bpInfo = collect_bp_info(runDir)
    files = dir(fullfile(runDir, 'bp_*.mat'));
    bpInfo = struct('idx', {}, 'filepath', {});
    for i = 1:numel(files)
        token = regexp(files(i).name, '^bp_(\d+)\.mat$', 'tokens', 'once');
        if isempty(token)
            continue;
        end
        bpInfo(end+1).idx = str2double(token{1}); %#ok<AGROW>
        bpInfo(end).filepath = fullfile(runDir, files(i).name);
    end
end

function idx = feasible_indices(runDir)
    idx = [];
    popFile = fullfile(runDir, 'final_popobj.mat');
    if ~exist(popFile, 'file')
        return;
    end
    data = load(popFile);
    if ~isfield(data, 'PopObj')
        return;
    end
    pop = data.PopObj;
    if ~isnumeric(pop) || ndims(pop) ~= 2
        return;
    end
    if size(pop, 2) ~= 4 && size(pop, 1) == 4
        pop = pop';
    end
    if size(pop, 2) < 4
        return;
    end
    finiteRows = all(isfinite(pop(:, 1:4)), 2);
    idx = find(finiteRows)';
end

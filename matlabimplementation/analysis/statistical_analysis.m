%% Research Analysis: Compute Metrics + Statistical Summary
% This script computes HV (if needed) and summarizes HV plus objective costs.

clear; clc;

% Initialize paths
scriptDir = fileparts(mfilename('fullpath'));
run(fullfile(scriptDir, '..', 'startup.m'));

% Speed/accuracy controls
hvSamples = 2000; % Reduce for speed (default HV.m uses 10000)
maxPoints = 100;  % Limit PopObj rows for faster HV
rng(0); % Reproducible subsampling

% Optional filters (leave empty to process all)
targetAlgorithms = {}; % e.g., {'NMOPSO'}
targetProblems = {};   % e.g., {'c_100'}
maxRuns = 0;           % 0 = all runs

resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');

% Identify Algorithm Folders
algoFolders = dir(resultsDir);
algoFolders = algoFolders([algoFolders.isdir]);
algoFolders = algoFolders(~strncmp({algoFolders.name}, '.', 1) & ~strcmp({algoFolders.name}, 'Plots'));

if isempty(algoFolders)
    error('No results found. Run scripts/run_benchmark.m first.');
end

% Pre-compute HV reference points per problem (across all algorithms/runs)
refPoints = struct();
fprintf('\nScanning results to set HV reference points...\n');
for a = 1:numel(algoFolders)
    algoDir = fullfile(resultsDir, algoFolders(a).name);
    runs = collect_run_dirs(algoDir);
    for r = 1:numel(runs)
        runDir = runs(r).runDir;
        folderName = runs(r).problemName;
        [PopObj, M] = load_run_popobj(runDir);
        PopObj = sanitize_popobj(PopObj, M);
        if isempty(PopObj)
            continue;
        end
        maxVals = max(PopObj, [], 1);
        if isfield(refPoints, folderName)
            refPoints.(folderName) = max(refPoints.(folderName), maxVals);
        else
            refPoints.(folderName) = maxVals;
        end
    end
end
refNames = fieldnames(refPoints);
for k = 1:numel(refNames)
    name = refNames{k};
    refPoint = refPoints.(name) * 1.1;
    refPoint(refPoint <= 0) = 1;
    refPoints.(name) = refPoint;
end

for a = 1:numel(algoFolders)
    algoName = algoFolders(a).name;
    algoDir = fullfile(resultsDir, algoName);

    if ~isempty(targetAlgorithms) && ~any(strcmpi(algoName, targetAlgorithms))
        continue;
    end

    [isAblation, variantDirs] = detect_ablation_layout(algoDir);
    if ~isAblation
        process_algo_variant(algoName, algoDir, refPoints, targetProblems, maxRuns, maxPoints, hvSamples);
    else
        for v = 1:numel(variantDirs)
            variantName = variantDirs(v).name;
            variantDir = fullfile(algoDir, variantName);
            displayName = sprintf('%s / %s', algoName, variantName);
            process_algo_variant(displayName, variantDir, refPoints, targetProblems, maxRuns, maxPoints, hvSamples);
        end
    end
end

function [isAblation, variantDirs] = detect_ablation_layout(algoDir)
    variantDirs = dir(algoDir);
    variantDirs = variantDirs([variantDirs.isdir]);
    variantDirs = variantDirs(~strncmp({variantDirs.name}, '.', 1));
    isAblation = true;
    hasDirectRuns = false;
    for i = 1:numel(variantDirs)
        runFolders = dir(fullfile(algoDir, variantDirs(i).name, 'Run_*'));
        if ~isempty(runFolders)
            hasDirectRuns = true;
            break;
        end
    end
    if hasDirectRuns
        isAblation = false;
    end
end

function process_algo_variant(displayName, baseDir, refPoints, targetProblems, maxRuns, maxPoints, hvSamples)
    fprintf('\n======================================================\n');
    fprintf('Statistical Summary for Algorithm: %s\n', displayName);
    fprintf('======================================================\n');
    fprintf('%-30s | %-15s | %-15s | %-15s | %-15s | %-15s\n', ...
        'Problem Scenario', 'HV', 'J1', 'J2', 'J3', 'J4');
    fprintf('%s\n', repmat('-', 1, 110));

    probFolders = dir(baseDir);
    probFolders = probFolders([probFolders.isdir]);
    probFolders = probFolders(~strncmp({probFolders.name}, '.', 1));

    resultsTable = table();

    for i = 1:numel(probFolders)
        folderName = probFolders(i).name;
        if ~isempty(targetProblems) && ~any(strcmpi(folderName, targetProblems))
            continue;
        end
        probDir = fullfile(baseDir, folderName);
        runFolders = dir(fullfile(probDir, 'Run_*'));
        runFolders = runFolders([runFolders.isdir]);

        if isempty(runFolders)
            continue;
        end

        if maxRuns > 0 && numel(runFolders) > maxRuns
            runFolders = runFolders(1:maxRuns);
        end

        hvScores = [];
        objMeans = [];
        problemIndex = 3;

        for r = 1:numel(runFolders)
            runDir = fullfile(probDir, runFolders(r).name);
            [PopObj, M, problemIndex] = load_run_popobj(runDir);
            PopObj = sanitize_popobj(PopObj, M);
            if isempty(PopObj)
                continue;
            end
            if maxPoints > 0 && size(PopObj, 1) > maxPoints
                idx = randperm(size(PopObj, 1), maxPoints);
                PopObj = PopObj(idx, :);
            end

            refPoint = [];
            if isfield(refPoints, folderName)
                refPoint = refPoints.(folderName);
            end
            try
                hv = calMetric(1, PopObj, problemIndex, M, hvSamples, refPoint);
            catch
                hv = NaN;
            end
            hvScores = [hvScores; hv];
            objMeans = [objMeans; mean(PopObj, 1)];
        end

        [meanHV, stdHV] = mean_std(hvScores, 1);
        [meanJ, stdJ] = mean_std(objMeans, 4);

        fprintf('%-30s | %6.4f ± %6.4f | %6.4f ± %6.4f | %6.4f ± %6.4f | %6.4f ± %6.4f | %6.4f ± %6.4f\n', ...
            folderName, meanHV, stdHV, ...
            meanJ(1), stdJ(1), meanJ(2), stdJ(2), meanJ(3), stdJ(3), meanJ(4), stdJ(4));

        cleanName = strrep(folderName, '_', '\\_');
        row = {cleanName, ...
               sprintf('$%.4f \\pm %.4f$', meanHV, stdHV), ...
               sprintf('$%.4f \\pm %.4f$', meanJ(1), stdJ(1)), ...
               sprintf('$%.4f \\pm %.4f$', meanJ(2), stdJ(2)), ...
               sprintf('$%.4f \\pm %.4f$', meanJ(3), stdJ(3)), ...
               sprintf('$%.4f \\pm %.4f$', meanJ(4), stdJ(4))};
        resultsTable = [resultsTable; row];

        if ~isempty(hvScores)
            bestScores = [hvScores, zeros(size(hvScores))];
            save(fullfile(probDir, 'final_hv.mat'), 'bestScores');
        end
    end

    fprintf('\n--- LaTeX Table Code for %s ---\n', displayName);
    fprintf('\\begin{table}[ht]\n');
    fprintf('\\centering\n');
    fprintf('\\caption{Performance of %s on UAV Benchmark}\n', displayName);
    fprintf('\\begin{tabular}{l|c|c|c|c|c}\n');
    fprintf('\\hline\n');
    fprintf('Scenario & HV & J1 & J2 & J3 & J4 \\\\ \\hline\n');
    for i = 1:size(resultsTable, 1)
        fprintf('%s & %s & %s & %s & %s & %s \\\\ \\hline\n', ...
            resultsTable{i,1}{1}, resultsTable{i,2}{1}, resultsTable{i,3}{1}, ...
            resultsTable{i,4}{1}, resultsTable{i,5}{1}, resultsTable{i,6}{1});
    end
    fprintf('\\hline\n');
    fprintf('\\end{tabular}\n');
    fprintf('\\end{table}\n');
end

function runs = collect_run_dirs(algoDir)
    runs = struct('runDir', {}, 'problemName', {});
    level1 = dir(algoDir);
    level1 = level1([level1.isdir]);
    level1 = level1(~strncmp({level1.name}, '.', 1));
    for i = 1:numel(level1)
        p1 = fullfile(algoDir, level1(i).name);
        runFolders = dir(fullfile(p1, 'Run_*'));
        runFolders = runFolders([runFolders.isdir]);
        if ~isempty(runFolders)
            for r = 1:numel(runFolders)
                runs(end+1).runDir = fullfile(p1, runFolders(r).name); %#ok<AGROW>
                runs(end).problemName = level1(i).name;
            end
            continue;
        end
        level2 = dir(p1);
        level2 = level2([level2.isdir]);
        level2 = level2(~strncmp({level2.name}, '.', 1));
        for j = 1:numel(level2)
            p2 = fullfile(p1, level2(j).name);
            runFolders = dir(fullfile(p2, 'Run_*'));
            runFolders = runFolders([runFolders.isdir]);
            for r = 1:numel(runFolders)
                runs(end+1).runDir = fullfile(p2, runFolders(r).name); %#ok<AGROW>
                runs(end).problemName = level2(j).name;
            end
        end
    end
end

function [PopObj, M, problemIndex] = load_run_popobj(runDir)
    popFile = fullfile(runDir, 'final_popobj.mat');
    PopObj = [];
    M = [];
    problemIndex = 3;
    if exist(popFile, 'file')
        data = load(popFile);
        if isfield(data, 'PopObj')
            PopObj = data.PopObj;
        end
        if isfield(data, 'M')
            M = data.M;
        end
        if isfield(data, 'problemIndex')
            problemIndex = data.problemIndex;
        end
    end
    if ~isempty(PopObj) && isempty(M)
        M = size(PopObj, 2);
    end
    if isempty(PopObj)
        bpFiles = dir(fullfile(runDir, 'bp_*.mat'));
        if isempty(bpFiles)
            return;
        end
        for b = 1:numel(bpFiles)
            bpData = load(fullfile(runDir, bpFiles(b).name));
            if isfield(bpData, 'dt_sv') && isfield(bpData.dt_sv, 'objs')
                PopObj = [PopObj; bpData.dt_sv.objs(:)'];
            end
        end
        if isempty(M) && ~isempty(PopObj)
            M = size(PopObj, 2);
        end
    end
    if ~isempty(PopObj) && ~isempty(M) && size(PopObj, 2) ~= M && size(PopObj, 1) == M
        PopObj = PopObj';
    end
end

function [meanVals, stdVals] = mean_std(values, expectedCols)
    if nargin < 2
        expectedCols = [];
    end
    if isempty(values)
        if isempty(expectedCols)
            meanVals = 0;
            stdVals = 0;
        else
            meanVals = zeros(1, expectedCols);
            stdVals = zeros(1, expectedCols);
        end
        return;
    end
    if isvector(values) && size(values, 2) == 1
        values = values(:);
    end
    if ~isempty(expectedCols) && size(values, 2) ~= expectedCols
        values = reshape(values, [], expectedCols);
    end
    meanVals = zeros(1, size(values, 2));
    stdVals = zeros(1, size(values, 2));
    for i = 1:size(values, 2)
        v = values(:, i);
        v = v(isfinite(v));
        if isempty(v)
            meanVals(i) = 0;
            stdVals(i) = 0;
        else
            meanVals(i) = mean(v);
            stdVals(i) = std(v);
        end
    end
end

function PopObj = sanitize_popobj(PopObj, M)
    if isempty(PopObj)
        return;
    end
    if ~isempty(M) && size(PopObj, 2) ~= M && size(PopObj, 1) == M
        PopObj = PopObj';
    end
    PopObj = PopObj(all(isfinite(PopObj), 2), :);
end

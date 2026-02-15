%% Research Analysis: Compute Metrics Offline
% This script computes HV/PD from saved objective sets after benchmarks finish.
clear; clc;

% Initialize paths
scriptDir = fileparts(mfilename('fullpath'));
run(fullfile(scriptDir, '..', 'startup.m'));

% Speed/accuracy controls
hvSamples = 2000; % Reduce for speed (default HV.m uses 10000)
maxPoints = 100;  % Limit PopObj rows for faster HV/PD (critical for PD)
rng(0); % Reproducible subsampling

% Optional filters (leave empty to process all)
targetAlgorithms = {}; % e.g., {'NMOPSO'}
targetProblems = {};   % e.g., {'c_100'}
maxRuns = 0;           % 0 = all runs

resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');

fprintf('Starting offline metric computation at %s\n', datestr(now));
fprintf('HV samples: %d | Max points: %d\n', hvSamples, maxPoints);
drawnow;

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
    probFolders = dir(algoDir);
    probFolders = probFolders([probFolders.isdir]);
    probFolders = probFolders(~strncmp({probFolders.name}, '.', 1));
    for i = 1:numel(probFolders)
        folderName = probFolders(i).name;
        probDir = fullfile(algoDir, folderName);
        runFolders = dir(fullfile(probDir, 'Run_*'));
        runFolders = runFolders([runFolders.isdir]);
        for r = 1:numel(runFolders)
            runDir = fullfile(probDir, runFolders(r).name);
            popFile = fullfile(runDir, 'final_popobj.mat');
            PopObj = [];
            M = [];
            if exist(popFile, 'file')
                data = load(popFile);
                if isfield(data, 'PopObj')
                    PopObj = data.PopObj;
                end
                if isfield(data, 'M')
                    M = data.M;
                end
            end
            if ~isempty(PopObj) && isempty(M)
                M = size(PopObj, 2);
            end
            if isempty(PopObj)
                bpFiles = dir(fullfile(runDir, 'bp_*.mat'));
                if ~isempty(bpFiles)
                    PopObj = [];
                    for b = 1:numel(bpFiles)
                        bpData = load(fullfile(runDir, bpFiles(b).name));
                        if isfield(bpData, 'dt_sv') && isfield(bpData.dt_sv, 'objs')
                            PopObj = [PopObj; bpData.dt_sv.objs(:)'];
                        end
                    end
                    if isempty(M)
                        M = size(PopObj, 2);
                    end
                end
            end
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
    fprintf('\nComputing metrics for Algorithm: %s\n', algoName);
    drawnow;

    if ~isempty(targetAlgorithms) && ~any(strcmpi(algoName, targetAlgorithms))
        fprintf('  - Skipping (filtered)\n');
        drawnow;
        continue;
    end

    % Identify Problem Folders for this Algorithm
    probFolders = dir(algoDir);
    probFolders = probFolders([probFolders.isdir]);
    probFolders = probFolders(~strncmp({probFolders.name}, '.', 1));

    fprintf('  - Found %d problem folders\n', numel(probFolders));
    drawnow;

    for i = 1:numel(probFolders)
        folderName = probFolders(i).name;
        if ~isempty(targetProblems) && ~any(strcmpi(folderName, targetProblems))
            continue;
        end
        probDir = fullfile(algoDir, folderName);
        runFolders = dir(fullfile(probDir, 'Run_*'));
        runFolders = runFolders([runFolders.isdir]);

        if isempty(runFolders)
            fprintf('  - %s: no runs found, skipping.\n', folderName);
            drawnow;
            continue;
        end

        if maxRuns > 0 && numel(runFolders) > maxRuns
            runFolders = runFolders(1:maxRuns);
        end

        fprintf('  - %s: %d runs\n', folderName, numel(runFolders));
        drawnow;
        problemStart = tic;

        bestScores = zeros(numel(runFolders), 2);
        scoreCount = 0;

        for r = 1:numel(runFolders)
            runDir = fullfile(probDir, runFolders(r).name);
            fprintf('    * Run %d/%d: %s\n', r, numel(runFolders), runFolders(r).name);
            drawnow;
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
                    fprintf('      - No bp_*.mat found, skipping.\n');
                    drawnow;
                    continue;
                end

                PopObj = [];
                for b = 1:numel(bpFiles)
                    bpData = load(fullfile(runDir, bpFiles(b).name));
                    if isfield(bpData, 'dt_sv') && isfield(bpData.dt_sv, 'objs')
                        PopObj = [PopObj; bpData.dt_sv.objs(:)'];
                    end
                end
                if isempty(M)
                    M = size(PopObj, 2);
                end
            end

            if isempty(PopObj)
                fprintf('      - No objective data found, skipping.\n');
                drawnow;
                continue;
            end

            if size(PopObj, 2) ~= M && size(PopObj, 1) == M
                PopObj = PopObj';
            end

            PopObj = sanitize_popobj(PopObj, M);
            if isempty(PopObj)
                fprintf('      - No valid objective data found, skipping.\n');
                drawnow;
                continue;
            end

            if maxPoints > 0 && size(PopObj, 1) > maxPoints
                idx = randperm(size(PopObj, 1), maxPoints);
                PopObj = PopObj(idx, :);
            end

            % Only calculate HV to prevent PD bottlenecks
            scoreCount = scoreCount + 1;
            refPoint = [];
            if isfield(refPoints, folderName)
                refPoint = refPoints.(folderName);
            end
            bestScores(scoreCount, :) = [calMetric(1, PopObj, problemIndex, M, hvSamples, refPoint), 0];
        end

        if scoreCount > 0
            bestScores = bestScores(1:scoreCount, :);
            save(fullfile(probDir, 'final_hv.mat'), 'bestScores');
            fprintf('  - %s: %d runs processed in %.2f seconds.\n', folderName, scoreCount, toc(problemStart));
            drawnow;
        else
            fprintf('  - %s: no objective data found, skipping.\n', folderName);
            drawnow;
        end
    end
end

fprintf('\nMetric computation complete.\n');
drawnow;

function PopObj = sanitize_popobj(PopObj, M)
    if isempty(PopObj)
        return;
    end
    if ~isempty(M) && size(PopObj, 2) ~= M && size(PopObj, 1) == M
        PopObj = PopObj';
    end
    PopObj = PopObj(all(isfinite(PopObj), 2), :);
end

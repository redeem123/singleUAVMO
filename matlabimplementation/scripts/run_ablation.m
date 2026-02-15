%% NMOPSO Ablation Study Runner
% Runs NMOPSO ablation variants across benchmark problems.

close all; clear; clc; format compact;

% Initialize paths
scriptDir = fileparts(mfilename('fullpath'));
run(fullfile(scriptDir, '..', 'startup.m'));

% Configuration
params.Generations = 200;
params.pop = 80;
params.Runs = 6;
% Metrics (HV/PD) are computed offline via analysis/compute_metrics.m.
params.computeMetrics = false;
% Parallel mode: 'parfeval' avoids parfor transparency errors on some setups.
params.useParallel = false;
params.parallelMode = 'parfor';
% Enable ablation study
params.ablationStudy = true;

% Base results directory for ablation
params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'NMOPSO_ABLATION');

% Safety clearance parameters for F2 (terrain/obstacle clearance)
params.safeDist = 20;
params.droneSize = 1;

% Get all benchmark problems
problemDir = fullfile(fileparts(mfilename('fullpath')), '..', 'problems');
problemFiles = dir(fullfile(problemDir, '*.mat'));

if ~isfolder(params.resultsDir)
    mkdir(params.resultsDir);
end

startTime = tic;
fprintf('Starting NMOPSO Ablation Suite at %s\n', datestr(now));
fprintf('-------------------------------------------\n');

for i = 1:(numel(problemFiles)-13)
    % Load Problem
    fileName = problemFiles(i).name;
    load(fullfile(problemDir, fileName), 'terrainStruct');
    terrainStruct.safeDist = params.safeDist;
    terrainStruct.droneSize = params.droneSize;

    % Extract clean problem name
    problemName = strrep(fileName, 'terrainStruct_', '');
    problemName = strrep(problemName, '.mat', '');
    params.problemName = problemName;
    params.problemIndex = i;

    fprintf('Problem (%d/%d): %s\n', i, numel(problemFiles), problemName);
    problemStart = tic;

    % --- Execute NMOPSO Ablations ---
    fprintf('>>> Running NMOPSO Ablation...\n');
    run_nmopso(terrainStruct, params);

    fprintf('  - Completed in %.2f seconds\n', toc(problemStart));
    fprintf('-------------------------------------------\n');
end

totalTime = toc(startTime);
fprintf('Ablation Suite Finished. Total time: %.2f seconds.\n', totalTime);

%% UAV Path Planning Benchmarking Driver
% This script automates the execution of multi-objective algorithms across all problems.

close all; clear; clc; format compact;

% Initialize paths
scriptDir = fileparts(mfilename('fullpath'));
run(fullfile(scriptDir, '..', 'startup.m'));

% Configuration
params.Generations = 500;
params.pop = 100;
params.Runs = 14;
% Metrics (HV/PD) are computed offline via analysis/compute_metrics.m.
params.computeMetrics = false;
% Parallel mode: 'parfeval' avoids parfor transparency errors on some setups.
params.useParallel = true;
params.parallelMode = 'parfeval';
params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
% Safety clearance parameters for F2 (terrain/obstacle clearance)
params.safeDist = 20;
params.droneSize = 1;

% TCOT-CTM-EA default configuration (used by run_ctmea).
params.ctm = struct();
params.ctm.scheduler = 'performance';
params.ctm.lambda0 = 0.0;
params.ctm.stepSize = 0.08;
params.ctm.stepSizeDown = 0.06;
params.ctm.progressDrift = 0.01;
params.ctm.feasHigh = 0.80;
params.ctm.feasLow = 0.35;
params.ctm.divThreshold = 0.08;

params.ctm.otInterval = 5;
params.ctm.transferFraction = 0.50;
params.ctm.sourcePoolSize = 120;
params.ctm.targetPoolSize = 120;
params.ctm.minArchiveForOT = 30;
params.ctm.archiveMaxSize = 500;
params.ctm.archiveInjectCount = 16;
params.ctm.minLambdaGap = 0.05;
params.ctm.minSourceEntries = 8;
params.ctm.sourceDistanceBias = 0.25;

params.ctm.alpha = 1.0;
params.ctm.beta0 = 1.2;
params.ctm.betaMin = 0.1;
params.ctm.betaMax = 8.0;
params.ctm.betaEta = 0.10;
params.ctm.gamma = 0.5;
params.ctm.otEpsilon = 0.08;
params.ctm.otMaxIter = 80;
params.ctm.otTol = 1e-4;

params.ctm.useCounterfactual = true;
params.ctm.counterfactualControlCount = 16;
params.ctm.transferBlendMin = 0.25;
params.ctm.transferBlendMax = 0.75;
params.ctm.transferNoise = 0.015;
params.ctm.signatureMaxObstacles = 4;
params.ctm.ntiTolerance = 1e-9;
params.ctm.objWeights = [1, 1, 1, 1];

% Get all benchmark problems
problemDir = fullfile(fileparts(mfilename('fullpath')), '..', 'problems');
problemFiles = dir(fullfile(problemDir, '*.mat'));

if ~isfolder(params.resultsDir)
    mkdir(params.resultsDir);
end

startTime = tic;
fprintf('Starting Benchmark Suite at %s\n', datestr(now));
fprintf('-------------------------------------------\n');

for i = 1:numel(problemFiles)
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
    
    %--- Execute NMOPSO ---
    fprintf('>>> Running NMOPSO...\n');
    params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'NMOPSO');
    if ~isfolder(params.resultsDir), mkdir(params.resultsDir); end
    run_nmopso(terrainStruct, params);

    % --- Execute MOPSO ---
    fprintf('>>> Running MOPSO...\n');
    params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'MOPSO');
    if ~isfolder(params.resultsDir), mkdir(params.resultsDir); end
    run_mopso(terrainStruct, params);

    % --- Execute NSGA-II ---
    fprintf('>>> Running NSGA-II...\n');
    params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'NSGA-II');
    if ~isfolder(params.resultsDir), mkdir(params.resultsDir); end
    run_nsga2(terrainStruct, params); 

    % --- Execute NSGA-III ---
    fprintf('>>> Running NSGA-III...\n');
    params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'NSGA-III');
    if ~isfolder(params.resultsDir), mkdir(params.resultsDir); end
    run_nsga3(terrainStruct, params);

    % --- Execute MO-MFEA ---
    fprintf('>>> Running MO-MFEA...\n');
    params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'MO-MFEA');
    if ~isfolder(params.resultsDir), mkdir(params.resultsDir); end
    run_momfea(terrainStruct, params);

    % --- Execute MO-MFEA-II ---
    fprintf('>>> Running MO-MFEA-II...\n');
    params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'MO-MFEA-II');
    if ~isfolder(params.resultsDir), mkdir(params.resultsDir); end
    run_momfea2(terrainStruct, params);

    % --- Execute CTM-EA ---
    fprintf('>>> Running CTM-EA...\n');
    params.resultsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'CTM-EA');
    if ~isfolder(params.resultsDir), mkdir(params.resultsDir); end
    run_ctmea(terrainStruct, params);
    
    fprintf('  - Completed in %.2f seconds\n', toc(problemStart));
    fprintf('-------------------------------------------\n');
end

totalTime = toc(startTime);
fprintf('Benchmark Suite Finished. Total time: %.2f seconds.\n', totalTime);

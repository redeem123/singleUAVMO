% Project Startup Script
% Run this script to initialize the workspace for UAV Path Planning research.

fprintf('Initializing UAV Path Planning Research Environment...\n');

% Get project root
projectRoot = fileparts(mfilename('fullpath'));

% Add necessary directories to path
addpath(fullfile(projectRoot, 'core'));
addpath(fullfile(projectRoot, 'core', 'metrics'));
addpath(fullfile(projectRoot, 'core', 'operators'));
addpath(fullfile(projectRoot, 'algorithms', 'NSGA-II'));
addpath(fullfile(projectRoot, 'algorithms', 'NSGA-III'));
addpath(fullfile(projectRoot, 'algorithms', 'NMOPSO'));
addpath(fullfile(projectRoot, 'algorithms', 'MOPSO'));
addpath(fullfile(projectRoot, 'algorithms', 'EMT'));
addpath(fullfile(projectRoot, 'algorithms', 'MO-MFEA'));
addpath(fullfile(projectRoot, 'algorithms', 'MO-MFEA-II'));
addpath(fullfile(projectRoot, 'algorithms', 'CTM-EA'));
addpath(fullfile(projectRoot, 'analysis'));
addpath(fullfile(projectRoot, 'problems'));

fprintf('Paths successfully initialized.\n');
fprintf('Available Scenarios: %d\n', numel(dir(fullfile(projectRoot, 'problems', '*.mat'))));
fprintf('Ready for benchmarking.\n');

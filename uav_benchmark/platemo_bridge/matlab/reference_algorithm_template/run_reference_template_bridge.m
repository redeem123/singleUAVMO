function run_reference_template_bridge(configPath)
% Template for MATLAB reference algorithms driven by the Python benchmark.
%
% Required config fields:
%   referenceRoot, bridgeRoot, workDir, outputPath, pythonExecutable,
%   pythonPath, contextPath, N, M, D, maxFE, lower, upper, seed.
%
% Replace the placeholder algorithm call with the official MATLAB algorithm.
% Keep operators/selection/archives in MATLAB, but route every objective or
% constraint evaluation through BridgeEvaluateInPython.m.

    cfg = load(configPath);
    addpath(genpath(char(cfg.referenceRoot)));
    addpath(char(cfg.bridgeRoot));
    rng(double(cfg.seed),'twister');
    setenv('PYTHON_BRIDGE_CONFIG',char(configPath));

    % For PlatEMO-style code, use PythonBridgeProblem or
    % PythonBridgeLegacyProblem. For path-native UAV code, shim the official
    % evaluation method and call BridgeEvaluatePathPopulation.
    error('ReferenceTemplate:NotImplemented','Replace this template with a comparator-specific bridge.');

    % Expected output variables:
    % PopDec = zeros(0,double(cfg.D)); %#ok<UNRCH,NASGU>
    % PopObj = zeros(0,double(cfg.M)); %#ok<NASGU>
    % PopCon = zeros(0,1); %#ok<NASGU>
    % FE = 0; %#ok<NASGU>
    % runtime = 0; %#ok<NASGU>
    % save(char(cfg.outputPath),'PopDec','PopObj','PopCon','FE','runtime');
end

function run_reference_legacy_bridge(configPath)
% Run an old PlatEMO GLOBAL-style reference algorithm on PythonBridgeLegacyProblem.

    cfg = load(configPath);
    addpath(genpath(char(cfg.platemoRoot)));
    addpath(char(cfg.bridgeRoot));
    addpath(genpath(char(cfg.referenceRoot)));
    if isfield(cfg,'fairShimRoot') && ~isempty(char(cfg.fairShimRoot))
        addpath(char(cfg.fairShimRoot));
    end
    addpath(fullfile(char(cfg.bridgeRoot),'legacy_shims'));
    rng(double(cfg.seed),'twister');
    setenv('PYTHON_BRIDGE_CONFIG',char(configPath));

    [~, ~, ~, ~, GlobalBest, runtime, ~, ~, ~, ~] = main( ...
        '-algorithm',str2func(char(cfg.algorithmFunction)), ...
        '-problem',@PythonBridgeLegacyProblem, ...
        '-N',double(cfg.N), ...
        '-M',double(cfg.M), ...
        '-D',double(cfg.D), ...
        '-evaluation',double(cfg.maxFE), ...
        '-run',1);

    if isempty(GlobalBest.result)
        PopDec = zeros(0,double(cfg.D));
        PopObj = zeros(0,double(cfg.M));
        PopCon = zeros(0,1);
    else
        Population = GlobalBest.result{end};
        PopDec = Population.decs;
        PopObj = Population.objs;
        PopCon = Population.cons;
    end
    FE = GlobalBest.evaluated; %#ok<NASGU>
    save(char(cfg.outputPath),'PopDec','PopObj','PopCon','FE','runtime');
end

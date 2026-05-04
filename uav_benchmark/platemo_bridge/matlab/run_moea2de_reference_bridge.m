function run_moea2de_reference_bridge(configPath)
% Run the official MOEA-2DE MATLAB code and export its final paths.

    cfg = load(configPath);
    addpath(genpath(char(cfg.referenceRoot)));
    addpath(char(cfg.fairShimRoot));
    rng(double(cfg.seed),'twister');
    setenv('PYTHON_BRIDGE_CONFIG',char(configPath));

    model = cfg.model;
    model.n = double(cfg.pathNodeCount);
    model.start = double(cfg.start);
    model.end = double(cfg.goal);

    [~,BestPopulation] = MOEA_2DE( ...
        double(cfg.M), ...
        model, ...
        double(cfg.N), ...
        double(cfg.generations), ...
        double(cfg.F), ...
        double(cfg.pc), ...
        double(cfg.pm), ...
        double(cfg.pathNodeCount), ...
        double(cfg.problemIndexForReference));

    if isempty(BestPopulation)
        PathStack = zeros(0,0,3);
        PopObj = zeros(0,double(cfg.M));
    else
        pathLengths = arrayfun(@(item) size(item.path,1),BestPopulation);
        maxLen = max(pathLengths);
        PathStack = nan(length(BestPopulation),maxLen,3);
        PopObj = zeros(length(BestPopulation),double(cfg.M));
        for i = 1 : length(BestPopulation)
            path = double(BestPopulation(i).path);
            PathStack(i,1:size(path,1),:) = path(:,1:3);
            PopObj(i,:) = double(BestPopulation(i).objs);
        end
    end
    save(char(cfg.outputPath),'PathStack','PopObj');
end

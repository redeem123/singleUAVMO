function Population = BridgeEvaluatePathPopulation(Population, configPath)
% Batch-evaluate a MATLAB UAV path population through the Python benchmark.

    if nargin < 2 || isempty(configPath)
        configPath = getenv('PYTHON_BRIDGE_CONFIG');
    end
    if isempty(configPath)
        error('BridgeEvaluatePathPopulation:MissingConfig','PYTHON_BRIDGE_CONFIG is not set.');
    end
    if isempty(Population)
        return;
    end

    pathLengths = arrayfun(@(item) size(item.path,1),Population);
    maxLen = max(pathLengths);
    PathStack = nan(length(Population),maxLen,3);
    for i = 1 : length(Population)
        path = double(Population(i).path(:,1:3));
        PathStack(i,1:size(path,1),:) = path;
    end

    [PopObj, PopCon] = BridgeEvaluateInPython(configPath,struct('PathStack',PathStack));
    for i = 1 : length(Population)
        Population(i).objs = double(PopObj(i,:));
        Population(i).cons = double(PopCon(i,:));
    end
end

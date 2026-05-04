function run_platemo_bridge(configPath)
% Run a PlatEMO algorithm on PythonBridgeProblem and save final decisions.

    cfg = load(configPath);
    addpath(genpath(char(cfg.platemoRoot)));
    addpath(char(cfg.bridgeRoot));
    rng(double(cfg.seed),'twister');

    Problem = PythonBridgeProblem('parameter',{configPath});
    Algorithm = feval(char(cfg.algorithmClass),'save',1,'outputFcn',@SilentOutput);
    Algorithm.Solve(Problem);

    if isempty(Algorithm.result)
        PopDec = zeros(0,Problem.D);
        PopObj = zeros(0,Problem.M);
        PopCon = zeros(0,1);
    else
        Population = Algorithm.result{end,2};
        PopDec = Population.decs;
        PopObj = Population.objs;
        PopCon = Population.cons;
    end
    FE = Problem.FE; %#ok<NASGU>
    runtime = Algorithm.metric.runtime; %#ok<NASGU>
    save(char(cfg.outputPath),'PopDec','PopObj','PopCon','FE','runtime');
end

function SilentOutput(~,~)
end

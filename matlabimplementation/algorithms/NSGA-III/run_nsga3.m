function [bestScores, gen_hv] = run_nsga3(model, params)
    % run_nsga3: NSGA-III implementation for UAV Path Planning
    % Inputs:
    %   model - terrainStruct with environment data
    %   params - struct with algorithm parameters (pop, Generations, Runs, resultsDir, problemName)

    Generations = params.Generations;
    pop = params.pop;
    M = 4; % Objectives
    problemIndex = 3;
    if isfield(params, 'problemIndex')
        problemIndex = params.problemIndex;
    end

    % Override model resolution for better agility
    model.n = 10;

    % Boundary definition
    MinValue = [model.xmin, model.ymin, model.zmin];
    MaxValue = [model.xmax, model.ymax, model.zmax];
    Boundary = [MaxValue; MinValue];

    resultsPath = fullfile(params.resultsDir, params.problemName);
    if ~isfolder(resultsPath)
        mkdir(resultsPath);
    end

    computeMetrics = false;
    if isfield(params, 'computeMetrics')
        computeMetrics = logical(params.computeMetrics);
    end

    useParallel = true;
    if isfield(params, 'useParallel')
        useParallel = logical(params.useParallel);
    end

    parallelMode = 'parfor';
    if isfield(params, 'parallelMode')
        parallelMode = params.parallelMode;
    end

    refPointMethod = '';
    if isfield(params, 'refPointMethod')
        refPointMethod = params.refPointMethod;
    end

    if isempty(refPointMethod)
        [Z, popRef] = UniformPoint(pop, M);
    else
        [Z, popRef] = UniformPoint(pop, M, refPointMethod);
    end

    if popRef ~= pop
        fprintf('NSGA-III: Adjusting population size from %d to %d to match reference points.\n', pop, popRef);
        pop = popRef;
    end
    params.pop = pop;

    if computeMetrics
        runScores = zeros(params.Runs, 2);
    else
        runScores = [];
    end

    if useParallel
        if strcmpi(parallelMode, 'parfeval')
            pool = gcp('nocreate');
            if isempty(pool)
                parpool;
            end
            futures(params.Runs, 1) = parallel.FevalFuture;
            for run = 1:params.Runs
                futures(run) = parfeval(@nsga3_single_run, 1, model, params, run, ...
                    computeMetrics, problemIndex, resultsPath, M, Boundary, Z);
            end
            for idx = 1:params.Runs
                [runIdx, score] = fetchNext(futures);
                if computeMetrics && ~isempty(score)
                    runScores(runIdx, :) = score;
                end
            end
        else
            if computeMetrics
                parfor run = 1:params.Runs
                    runScores(run, :) = nsga3_single_run(model, params, run, ...
                        computeMetrics, problemIndex, resultsPath, M, Boundary, Z);
                end
            else
                parfor run = 1:params.Runs
                    nsga3_single_run(model, params, run, computeMetrics, ...
                        problemIndex, resultsPath, M, Boundary, Z);
                end
            end
        end
    else
        for run = 1:params.Runs
            score = nsga3_single_run(model, params, run, computeMetrics, ...
                problemIndex, resultsPath, M, Boundary, Z);
            if computeMetrics && ~isempty(score)
                runScores(run, :) = score;
            end
        end
    end

    bestScores = runScores;
    if computeMetrics
        save(fullfile(resultsPath, 'final_hv.mat'), 'bestScores');
    end
end

function runScore = nsga3_single_run(model, params, run, computeMetrics, ...
    problemIndex, resultsPath, M, Boundary, Z)

    Generations = params.Generations;
    pop = params.pop;

    fprintf('  - Starting Run %d/%d\n', run, params.Runs);
    runStart = tic;

    current_population = Chromosome.empty(0, pop);
    for i = 1:pop
        p = Chromosome(model);
        p = initialize(p, model);
        p = evaluate(p, model);
        current_population(i) = p;
    end

    [PopObj, PopCon] = extract_pop_data(current_population, M);
    Zmin = compute_zmin(PopObj, PopCon);

    local_gen_hv = [];
    if computeMetrics
        local_gen_hv = zeros(Generations, 2);
    end

    for gen = 1:Generations
        if mod(gen, 100) == 0 || gen == 1
            fprintf('    - Run %d: Generation %d/%d\n', run, gen, Generations);
        end

        [~, PopCon] = extract_pop_data(current_population, M);
        cv = extract_cv(PopCon, pop);
        MatingPool = TournamentSelection(2, pop, cv);
        offspring = F_operator(current_population, MatingPool', Boundary, model);

        [OffObj, OffCon] = extract_pop_data(offspring, M);
        Zmin = update_zmin(Zmin, OffObj, OffCon);
        current_population = EnvironmentalSelection_NSGAIII([current_population, offspring], pop, Z, Zmin);

        if computeMetrics
            obj = [current_population.objs];
            PopObj = reshape(obj, M, length(current_population))';

            if mod(gen, 50) == 0 || gen == 1 || gen == Generations
                local_gen_hv(gen, :) = [calMetric(1, PopObj, problemIndex, M), ...
                    calMetric(2, PopObj, problemIndex, M)];
            elseif gen > 1
                local_gen_hv(gen, :) = local_gen_hv(gen-1, :);
            end
        end
    end

    run_dir = fullfile(resultsPath, sprintf('Run_%d', run));
    if ~isfolder(run_dir)
        mkdir(run_dir);
    end

    if computeMetrics
        save_data(fullfile(run_dir, 'gen_hv.mat'), local_gen_hv);
    end

    obj = [current_population.objs];
    PopObj = reshape(obj, M, length(current_population))';
    save(fullfile(run_dir, 'final_popobj.mat'), 'PopObj', 'problemIndex', 'M');
    runtimeSec = toc(runStart);
    feasibleCount = sum(all(isfinite(PopObj), 2));
    solutionCount = size(PopObj, 1);
    save(fullfile(run_dir, 'run_stats.mat'), 'runtimeSec', 'feasibleCount', 'solutionCount');
    if computeMetrics
        runScore = [calMetric(1, PopObj, problemIndex, M), ...
            calMetric(2, PopObj, problemIndex, M)];
    else
        runScore = [];
    end

    for i = 1:size(current_population, 2)
        local_dt_sv = struct();
        local_dt_sv.path = current_population(i).path;
        local_dt_sv.objs = current_population(i).objs;
        save_data(fullfile(run_dir, sprintf('bp_%d.mat', i)), local_dt_sv);
    end
end

function [PopObj, PopCon] = extract_pop_data(Population, M)
    PopSize = numel(Population);
    if PopSize == 0
        PopObj = zeros(0, M);
        PopCon = [];
        return;
    end
    obj = [Population.objs];
    PopObj = reshape(obj, M, PopSize)';

    consCell = {Population.cons};
    if isempty(consCell) || all(cellfun(@isempty, consCell))
        PopCon = [];
        return;
    end
    maxLen = max(cellfun(@numel, consCell));
    PopCon = zeros(PopSize, maxLen);
    for i = 1:PopSize
        c = consCell{i};
        if isempty(c)
            continue;
        end
        c = c(:)';
        PopCon(i, 1:numel(c)) = c;
    end
end

function cv = extract_cv(PopCon, pop)
    if isempty(PopCon)
        cv = zeros(pop, 1);
    else
        cv = sum(max(0, PopCon), 2);
    end
end

function Zmin = compute_zmin(PopObj, PopCon)
    if isempty(PopObj)
        Zmin = [];
        return;
    end
    feasible = feasible_mask(PopCon, size(PopObj, 1));
    if any(feasible)
        Zmin = min(PopObj(feasible, :), [], 1);
    else
        Zmin = min(PopObj, [], 1);
    end
end

function Zmin = update_zmin(Zmin, PopObj, PopCon)
    if isempty(PopObj)
        return;
    end
    feasible = feasible_mask(PopCon, size(PopObj, 1));
    if any(feasible)
        if isempty(Zmin)
            Zmin = min(PopObj(feasible, :), [], 1);
        else
            Zmin = min([Zmin; PopObj(feasible, :)], [], 1);
        end
    elseif isempty(Zmin)
        Zmin = min(PopObj, [], 1);
    end
end

function feasible = feasible_mask(PopCon, popSize)
    if isempty(PopCon)
        feasible = true(popSize, 1);
    else
        feasible = all(PopCon <= 0, 2);
    end
end

function save_data(filename, data)
    % Helper to save data inside parfor
    if isstruct(data)
        dt_sv = data;
        save(filename, 'dt_sv');
    else
        gen_hv = data;
        save(filename, 'gen_hv');
    end
end

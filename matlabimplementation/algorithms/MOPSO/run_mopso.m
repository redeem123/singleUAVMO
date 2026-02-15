function [bestScores, gen_hv] = run_mopso(model, params)
    % run_mopso: Baseline MOPSO implementation for UAV Path Planning

    Generations = params.Generations;
    pop = params.pop;
    M = 4;
    problemIndex = 3;
    if isfield(params, 'problemIndex')
        problemIndex = params.problemIndex;
    end

    % Override model resolution for better agility
    model.n = 10;

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
                futures(run) = parfeval(@mopso_single_run, 1, model, params, run, ...
                    computeMetrics, problemIndex, resultsPath, M);
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
                    runScores(run, :) = mopso_single_run(model, params, run, ...
                        computeMetrics, problemIndex, resultsPath, M);
                end
            else
                parfor run = 1:params.Runs
                    mopso_single_run(model, params, run, computeMetrics, ...
                        problemIndex, resultsPath, M);
                end
            end
        end
    else
        for run = 1:params.Runs
            score = mopso_single_run(model, params, run, computeMetrics, ...
                problemIndex, resultsPath, M);
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

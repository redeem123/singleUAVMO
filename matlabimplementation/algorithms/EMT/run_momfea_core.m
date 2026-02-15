function [bestScores, gen_hv] = run_momfea_core(model, params, algorithmName)
% run_momfea_core - Shared runner for MOMFEA and MOMFEAII via PlatEMO.

    M = 4;
    problemIndex = 3;
    if isfield(params, 'problemIndex')
        problemIndex = params.problemIndex;
    end

    if isfield(model, 'n') && ~isempty(model.n)
        model.n = 10;
    else
        model.n = 10;
    end
    nControl = model.n;

    auxModel = build_aux_model(model, params);

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
                futures(run) = parfeval(@momfea_single_run, 1, model, auxModel, params, run, ...
                    computeMetrics, problemIndex, resultsPath, M, algorithmName, nControl);
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
                    runScores(run, :) = momfea_single_run(model, auxModel, params, run, ...
                        computeMetrics, problemIndex, resultsPath, M, algorithmName, nControl);
                end
            else
                parfor run = 1:params.Runs
                    momfea_single_run(model, auxModel, params, run, computeMetrics, ...
                        problemIndex, resultsPath, M, algorithmName, nControl);
                end
            end
        end
    else
        for run = 1:params.Runs
            score = momfea_single_run(model, auxModel, params, run, computeMetrics, ...
                problemIndex, resultsPath, M, algorithmName, nControl);
            if computeMetrics && ~isempty(score)
                runScores(run, :) = score;
            end
        end
    end

    bestScores = runScores;
    gen_hv = [];
    if computeMetrics
        save(fullfile(resultsPath, 'final_hv.mat'), 'bestScores');
    end
end

function runScore = momfea_single_run(model, auxModel, params, run, computeMetrics, ...
    problemIndex, resultsPath, M, algorithmName, nControl)

    fprintf('  - Starting Run %d/%d\n', run, params.Runs);

    projectRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));

    maxFE = params.pop * (params.Generations + 1);
    if isfield(params, 'maxFE') && isnumeric(params.maxFE) && isfinite(params.maxFE) && params.maxFE > 0
        maxFE = round(params.maxFE);
    end

    baseSeed = 10000 + 1000 * problemIndex + run;
    if isfield(params, 'seed') && isnumeric(params.seed) && isfinite(params.seed)
        baseSeed = round(params.seed) + run;
    end

    [Dec, Obj] = run_platemo_once(projectRoot, algorithmName, model, auxModel, ...
        nControl, params.pop, maxFE, baseSeed, run, params);

    [PopDec, PopObj] = select_target_task(Dec, Obj, M);

    runDir = fullfile(resultsPath, sprintf('Run_%d', run));
    if ~isfolder(runDir)
        mkdir(runDir);
    end

    save(fullfile(runDir, 'final_popobj.mat'), 'PopObj', 'problemIndex', 'M');

    for i = 1:size(PopObj, 1)
        dt_sv = struct();
        dt_sv.path = decode_uav_path(PopDec(i, 1:end-1), model, nControl);
        dt_sv.objs = PopObj(i, :);
        save_data(fullfile(runDir, sprintf('bp_%d.mat', i)), dt_sv);
    end

    if computeMetrics
        runScore = [calMetric(1, PopObj, problemIndex, M), calMetric(2, PopObj, problemIndex, M)];
    else
        runScore = [];
    end
end

function [Dec, Obj] = run_platemo_once(projectRoot, algorithmName, model, auxModel, ...
    nControl, popSize, maxFE, seed, runId, params)

    oldPath = path;
    oldDir = pwd;
    cleanupPath = onCleanup(@() path(oldPath)); %#ok<NASGU>
    cleanupDir = onCleanup(@() cd(oldDir)); %#ok<NASGU>

    platemoRoot = fullfile(projectRoot, 'reference_code', 'PlatEMO', 'PlatEMO');
    addpath(genpath(platemoRoot));
    addpath(fullfile(projectRoot, 'algorithms', 'EMT'));

    rng(seed, 'twister');

    problemSpec = struct('models', {{auxModel, model}}, 'nControl', nControl);

    switch upper(algorithmName)
        case 'MOMFEA'
            rmp = 1;
            if isfield(params, 'mfeaRMP') && isnumeric(params.mfeaRMP) && isfinite(params.mfeaRMP)
                rmp = params.mfeaRMP;
            end
            algorithmArg = {@MOMFEA, rmp};
        case 'MOMFEAII'
            algorithmArg = @MOMFEAII;
        otherwise
            error('Unsupported algorithmName: %s', algorithmName);
    end

    [Dec, Obj, ~] = platemo( ...
        'algorithm', algorithmArg, ...
        'problem', {@UAV_MTOP, problemSpec}, ...
        'N', popSize, ...
        'maxFE', maxFE, ...
        'save', 0, ...
        'run', runId);
end

function [PopDec, PopObj] = select_target_task(Dec, Obj, M)
    penaltyValue = 1e6;
    penaltyTol = 1;

    if isempty(Dec) || isempty(Obj)
        PopDec = zeros(1, M + 1);
        PopObj = inf(1, M);
        return;
    end

    taskId = round(Dec(:, end));
    targetMask = taskId == 2;
    if ~any(targetMask)
        targetMask = true(size(taskId));
    end

    PopDec = Dec(targetMask, :);
    PopObj = Obj(targetMask, 1:min(M, size(Obj, 2)));

    if size(PopObj, 2) < M
        PopObj = [PopObj, inf(size(PopObj, 1), M - size(PopObj, 2))];
    end

    % PlatEMO task adapter uses 1e6 finite penalties for infeasible solutions.
    % Convert them back to Inf in exported benchmark outputs for fairness with
    % NMOPSO/MOPSO/NSGA/CTM pipelines that keep infeasible values as Inf.
    penaltyRows = all(PopObj >= (penaltyValue - penaltyTol), 2);
    PopObj(penaltyRows, :) = inf;

    if isempty(PopObj)
        PopDec = Dec(1, :);
        PopObj = inf(1, M);
    end
end

function auxModel = build_aux_model(model, params)
    auxModel = model;

    safeDistScale = 0.5;
    if isfield(params, 'mfeaAuxSafeDistScale') && isnumeric(params.mfeaAuxSafeDistScale) ...
            && isfinite(params.mfeaAuxSafeDistScale) && params.mfeaAuxSafeDistScale > 0
        safeDistScale = params.mfeaAuxSafeDistScale;
    end

    if isfield(auxModel, 'safeDist') && ~isempty(auxModel.safeDist)
        auxModel.safeDist = max(1, double(auxModel.safeDist) * safeDistScale);
    elseif isfield(auxModel, 'safe_dist') && ~isempty(auxModel.safe_dist)
        auxModel.safe_dist = max(1, double(auxModel.safe_dist) * safeDistScale);
    else
        auxModel.safeDist = 10;
    end

    noFlyScale = 0.8;
    if isfield(params, 'mfeaAuxNoFlyScale') && isnumeric(params.mfeaAuxNoFlyScale) ...
            && isfinite(params.mfeaAuxNoFlyScale) && params.mfeaAuxNoFlyScale > 0
        noFlyScale = params.mfeaAuxNoFlyScale;
    end

    if isfield(auxModel, 'nofly_r') && ~isempty(auxModel.nofly_r)
        auxModel.nofly_r = double(auxModel.nofly_r) * noFlyScale;
    end
end

function save_data(filename, data)
    dt_sv = data;
    save(filename, 'dt_sv');
end

function path = decode_uav_path(normDec, model, nControl)
    nControl = max(3, round(nControl));
    nMid = nControl - 2;
    needed = 3 * nMid;

    normDec = normDec(:)';
    if numel(normDec) < needed
        normDec = [normDec, 0.5 * ones(1, needed - numel(normDec))];
    elseif numel(normDec) > needed
        normDec = normDec(1:needed);
    end

    mid = reshape(normDec, [nMid, 3]);
    mid = max(0, min(1, mid));

    x = double(model.xmin) + mid(:, 1) * (double(model.xmax) - double(model.xmin));
    y = double(model.ymin) + mid(:, 2) * (double(model.ymax) - double(model.ymin));
    zAlpha = mid(:, 3);

    [x, idx] = sort(x, 'ascend');
    y = y(idx);
    zAlpha = zAlpha(idx);

    safeH = 0;
    if isfield(model, 'safeH') && ~isempty(model.safeH)
        safeH = double(model.safeH);
    end

    z = zeros(nMid, 1);
    for i = 1:nMid
        xi = max(1, min(size(model.H, 2), round(x(i))));
        yi = max(1, min(size(model.H, 1), round(y(i))));
        ground = double(model.H(yi, xi));

        minZ = max(double(model.zmin), ground + safeH);
        maxZ = double(model.zmax);
        if maxZ <= minZ
            z(i) = minZ;
        else
            z(i) = minZ + zAlpha(i) * (maxZ - minZ);
        end
    end

    path = zeros(nControl, 3);
    path(1, :) = double(model.start(:))';
    path(end, :) = double(model.end(:))';
    path(2:end-1, :) = [x, y, z];

    path(:, 1) = max(double(model.xmin), min(double(model.xmax), path(:, 1)));
    path(:, 2) = max(double(model.ymin), min(double(model.ymax), path(:, 2)));
    path(:, 3) = max(double(model.zmin), min(double(model.zmax), path(:, 3)));
end

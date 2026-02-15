function [bestScores, gen_hv] = run_ctmea(model, params)
% run_ctmea: TCOT-CTM-EA for UAV path planning.
%
% This implementation aligns CTM-EA with the TCOT-CTM-EA idea:
% 1) Continuous task navigation via lambda-manifold interpolation.
% 2) Topology-constrained entropic OT transfer.
% 3) Counterfactual transfer credit for online beta adaptation.

    M = 4;
    problemIndex = 3;
    if isfield(params, 'problemIndex')
        problemIndex = params.problemIndex;
    end

    % Keep consistency with other algorithms in the benchmark.
    model.n = 10;

    options = parse_tcot_options(params);
    easyModel = build_easy_model(model, options);

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
                futures(run) = parfeval(@tcot_single_run, 1, model, easyModel, params, options, run, ...
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
                    runScores(run, :) = tcot_single_run(model, easyModel, params, options, run, ...
                        computeMetrics, problemIndex, resultsPath, M);
                end
            else
                parfor run = 1:params.Runs
                    tcot_single_run(model, easyModel, params, options, run, ...
                        computeMetrics, problemIndex, resultsPath, M);
                end
            end
        end
    else
        for run = 1:params.Runs
            score = tcot_single_run(model, easyModel, params, options, run, ...
                computeMetrics, problemIndex, resultsPath, M);
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

function runScore = tcot_single_run(modelHard, modelEasy, params, options, run, ...
    computeMetrics, problemIndex, resultsPath, M)

    pop = params.pop;
    generations = params.Generations;

    fprintf('  - Starting Run %d/%d\n', run, params.Runs);

    boundaryHard = [modelHard.xmax, modelHard.ymax, modelHard.zmax; ...
                    modelHard.xmin, modelHard.ymin, modelHard.zmin];

    lambda = options.lambda0;
    beta = options.beta0;
    currentModel = interpolate_models(modelEasy, modelHard, lambda);
    currentTheta = extract_theta_vector(currentModel, modelHard, options);

    population = Chromosome.empty(0, pop);
    for i = 1:pop
        p = Chromosome(currentModel);
        p = initialize(p, currentModel);
        p = repair_chromosome(p, currentModel, options, true);
        population(i) = p;
    end

    [population, frontNo, crowdDis] = EnvironmentalSelection(population, pop, pop, M);

    history = initialize_history(generations, options, beta);
    archive = initialize_archive(options);
    archive = append_archive_from_population(archive, population, currentModel, currentTheta, ...
        lambda, options, frontNo, 0);

    local_gen_hv = [];
    if computeMetrics
        local_gen_hv = zeros(generations, 2);
    end

    for gen = 1:generations
        if mod(gen, 100) == 0 || gen == 1
            fprintf('    - Run %d: Generation %d/%d\n', run, gen, generations);
        end

        phaseProgress = gen / max(1, generations);
        inFeasibilityPhase = phaseProgress <= options.feasibilityPhaseRatio;

        state = build_scheduler_state(population, lambda, history, gen, generations);
        lambda = schedule_lambda(lambda, state, options);
        if inFeasibilityPhase
            lambda = min(lambda, options.feasibilityLambdaCap);
        end
        currentModel = interpolate_models(modelEasy, modelHard, lambda);
        currentTheta = extract_theta_vector(currentModel, modelHard, options);

        % Re-evaluate on current manifold task.
        population = reevaluate_population(population, currentModel, options, inFeasibilityPhase);
        [population, frontNo, crowdDis] = EnvironmentalSelection(population, pop, pop, M);

        matingPool = TournamentSelection(2, pop, frontNo, -crowdDis);
        baseOffspring = F_operator(population, matingPool', boundaryHard, currentModel);
        baseOffspring = repair_population(baseOffspring, currentModel, options, inFeasibilityPhase);

        domainOffspring = generate_domain_guided_offspring( ...
            population, archive, currentModel, boundaryHard, options, inFeasibilityPhase);

        otInfo = default_ot_info(beta);
        transferOffspring = Chromosome.empty(0, 0);

        if mod(gen, options.otInterval) == 0 && numel(archive.items) >= options.minArchiveForOT && ...
                (~inFeasibilityPhase || options.enableOTInFeasibilityPhase)
            [transferOffspring, otInfo] = generate_ot_transfer_offspring( ...
                population, baseOffspring, archive, currentModel, currentTheta, lambda, ...
                beta, options, boundaryHard);
            if otInfo.used && options.useCounterfactual
                beta = update_beta(beta, otInfo.tau, options);
                otInfo.betaAfter = beta;
            end
        end

        merged = [population, baseOffspring, domainOffspring];
        if ~isempty(transferOffspring)
            merged = [merged, transferOffspring];
        end
        [population, frontNo, crowdDis] = EnvironmentalSelection(merged, pop, numel(merged), M);

        popObj = population_to_obj(population, M);
        feasibleRatio = mean(all(isfinite(popObj), 2));
        diversity = population_diversity(popObj);

        hvForLog = NaN;
        if mod(gen, options.logInterval) == 0 || gen == 1 || gen == generations
            hvForLog = safe_hv(popObj, problemIndex, M, options.hvSamplesLog);
        end

        history.generation(gen) = gen;
        history.lambda(gen) = lambda;
        history.feasibility(gen) = feasibleRatio;
        history.diversity(gen) = diversity;
        history.hypervolume(gen) = hvForLog;
        history.beta(gen) = beta;
        history.tau(gen) = otInfo.tau;
        history.tmtm(gen) = otInfo.tmtm;
        history.nti(gen) = otInfo.nti;
        history.dM(gen) = otInfo.dM;
        history.transferGain(gen) = otInfo.gainTransfer;
        history.controlGain(gen) = otInfo.gainNoTransfer;
        history.entropy(gen) = otInfo.entropy;
        history.transferPairs(gen) = otInfo.pairCount;
        history.transferUsed(gen) = otInfo.used;

        if computeMetrics
            if mod(gen, 50) == 0 || gen == 1 || gen == generations
                local_gen_hv(gen, :) = [calMetric(1, popObj, problemIndex, M), ...
                                        calMetric(2, popObj, problemIndex, M)];
            elseif gen > 1
                local_gen_hv(gen, :) = local_gen_hv(gen-1, :);
            end
        end

        archive = append_archive_from_population(archive, population, currentModel, currentTheta, ...
            lambda, options, frontNo, gen);
    end

    % Final hard-task adaptation before saving.
    population = finalize_on_hard_model(population, modelHard, pop, M, options, boundaryHard);
    PopObj = population_to_obj(population, M);

    runDir = fullfile(resultsPath, sprintf('Run_%d', run));
    if ~isfolder(runDir)
        mkdir(runDir);
    end

    if computeMetrics
        save_data(fullfile(runDir, 'gen_hv.mat'), local_gen_hv);
    end

    save(fullfile(runDir, 'final_popobj.mat'), 'PopObj', 'problemIndex', 'M');

    for i = 1:numel(population)
        dt_sv = struct();
        dt_sv.path = population(i).path;
        dt_sv.objs = population(i).objs;
        save_data(fullfile(runDir, sprintf('bp_%d.mat', i)), dt_sv);
    end

    ctm_history = history; %#ok<NASGU>
    tcot_history = history; %#ok<NASGU>
    save(fullfile(runDir, 'ctm_history.mat'), 'ctm_history');
    save(fullfile(runDir, 'tcot_history.mat'), 'tcot_history');

    if computeMetrics
        runScore = [calMetric(1, PopObj, problemIndex, M), ...
                    calMetric(2, PopObj, problemIndex, M)];
    else
        runScore = [];
    end
end

function [transferOffspring, info] = generate_ot_transfer_offspring(population, baseOffspring, ...
    archive, currentModel, currentTheta, currentLambda, beta, options, boundaryHard)

    info = default_ot_info(beta);
    transferOffspring = Chromosome.empty(0, 0);

    sourceEntries = select_source_entries(archive, currentLambda, currentTheta, options);
    if isempty(sourceEntries)
        return;
    end

    nTargets = min(options.targetPoolSize, numel(population));
    if nTargets <= 0
        return;
    end

    if nTargets < numel(population)
        targetIdx = randperm(numel(population), nTargets);
    else
        targetIdx = 1:numel(population);
    end
    targetPop = population(targetIdx);

    m = numel(sourceEntries);
    n = numel(targetPop);
    if m == 0 || n == 0
        return;
    end

    sigDim = signature_dimension(options);
    sourceSig = zeros(m, sigDim);
    targetSig = zeros(n, sigDim);
    sourceTheta = zeros(m, numel(currentTheta));
    qSource = zeros(m, 1);
    qTarget = zeros(n, 1);

    for i = 1:m
        sourceSig(i, :) = sourceEntries(i).signature;
        sourceTheta(i, :) = sourceEntries(i).theta;
        qSource(i) = objs_quality(sourceEntries(i).objs, options);
    end
    for j = 1:n
        targetSig(j, :) = topology_signature(targetPop(j).path, currentModel, options);
        qTarget(j) = chromosome_quality(targetPop(j), options);
    end

    dMVec = sqrt(sum((sourceTheta - currentTheta).^2, 2) / max(1, numel(currentTheta)));
    dMMatrix = repmat(dMVec, 1, n);

    deltaH = zeros(m, n);
    for i = 1:m
        for j = 1:n
            deltaH(i, j) = signature_distance(sourceSig(i, :), targetSig(j, :));
        end
    end

    qAll = [qSource; qTarget];
    qMin = min(qAll);
    qRange = max(qAll) - qMin;
    if qRange <= 0
        qRange = 1;
    end
    qSourceN = (qSource - qMin) ./ qRange;
    qTargetN = (qTarget - qMin) ./ qRange;
    deltaQ = abs(bsxfun(@minus, qSourceN, qTargetN'));

    C = options.alpha * (dMMatrix.^2) + beta * deltaH + options.gamma * deltaQ;
    if any(~isfinite(C(:)))
        finiteVals = C(isfinite(C));
        if isempty(finiteVals)
            return;
        end
        C(~isfinite(C)) = max(finiteVals) + 1;
    end

    a = ones(m, 1) / m;
    b = ones(n, 1) / n;
    [piMat, isValid] = sinkhorn_transport(C, a, b, options.otEpsilon, options.otMaxIter, options.otTol);
    if ~isValid
        return;
    end

    nTransfer = max(1, round(options.transferFraction * numel(population)));
    [srcPick, tgtPickLocal] = sample_pairs_from_coupling(piMat, nTransfer);
    tgtPick = targetIdx(tgtPickLocal);

    transferOffspring = Chromosome.empty(0, nTransfer);
    parentScoresTransfer = zeros(nTransfer, 1);
    childScores = zeros(nTransfer, 1);
    for k = 1:nTransfer
        srcEntry = sourceEntries(srcPick(k));
        tgt = population(tgtPick(k));
        child = build_transfer_child(srcEntry, tgt, currentModel, boundaryHard, options);
        transferOffspring(k) = child;
        parentScoresTransfer(k) = chromosome_quality(tgt, options);
        childScores(k) = chromosome_quality(child, options);
    end

    gainTransfer = mean(parentScoresTransfer - childScores);
    nti = mean((childScores - parentScoresTransfer) > options.ntiTolerance);

    controlCount = min([options.counterfactualControlCount, numel(baseOffspring), numel(population)]);
    if controlCount > 0
        baseIdx = randperm(numel(baseOffspring), controlCount);
        parentIdx = randperm(numel(population), controlCount);
        pBase = individual_quality_scores(population(parentIdx), options);
        cBase = individual_quality_scores(baseOffspring(baseIdx), options);
        gainNoTransfer = mean(pBase - cBase);
    else
        gainNoTransfer = 0;
    end

    tau = gainTransfer - gainNoTransfer;
    tmtm = sum(sum(piMat .* deltaH));
    dM = sum(sum(piMat .* dMMatrix));
    p = piMat(piMat > 0);
    entropyVal = -sum(p .* log(p));

    info.used = true;
    info.tau = tau;
    info.tmtm = tmtm;
    info.nti = nti;
    info.dM = dM;
    info.gainTransfer = gainTransfer;
    info.gainNoTransfer = gainNoTransfer;
    info.entropy = entropyVal;
    info.pairCount = nTransfer;
end

function offspring = generate_domain_guided_offspring(population, archive, model, boundary, options, aggressivePhase)
    nPop = numel(population);
    nTarget = round(options.domainGuidedFraction * nPop);
    if aggressivePhase
        nTarget = max(nTarget, round(options.domainGuidedFractionFeas * nPop));
    end
    if nTarget <= 0
        offspring = Chromosome.empty(0, 0);
        return;
    end

    guideEntries = struct('rnvec', {}, 'path', {}, 'objs', {}, 'theta', {}, ...
        'lambda', {}, 'signature', {}, 'age', {}, 'generation', {});
    if options.domainGuidedUseArchive && ~isempty(archive.items)
        guideEntries = archive.items;
        if options.feasibleArchiveOnly
            feasibleMask = false(1, numel(guideEntries));
            for i = 1:numel(guideEntries)
                feasibleMask(i) = all(isfinite(guideEntries(i).objs));
            end
            if any(feasibleMask)
                guideEntries = guideEntries(feasibleMask);
            end
        end
    end

    offspring = Chromosome.empty(0, nTarget);
    for k = 1:nTarget
        parent = population(randi(nPop));
        bestChild = parent;
        bestCV = inf;
        bestQ = inf;

        for trial = 1:options.domainMutationTrials
            child = parent;
            if ~isempty(guideEntries)
                guide = guideEntries(randi(numel(guideEntries)));
                blend = options.domainGuidedBlend + 0.1 * randn;
                blend = max(0, min(1, blend));
                rn = (1 - blend) * double(parent.rnvec) + blend * double(guide.rnvec);
            else
                rn = double(parent.rnvec);
            end

            localNoise = options.domainGuidedNoise;
            if aggressivePhase
                localNoise = localNoise * options.feasibilityNoiseBoost;
            end
            rn = perturb_and_project_rnvec(rn, model, boundary, localNoise);
            child.rnvec = rn;
            child = repair_chromosome(child, model, options, aggressivePhase);

            cv = individual_constraint_value(child);
            q = chromosome_quality(child, options);
            if cv < bestCV || (abs(cv - bestCV) < 1e-12 && q < bestQ)
                bestChild = child;
                bestCV = cv;
                bestQ = q;
            end
        end
        offspring(k) = bestChild;
    end
end

function child = build_transfer_child(srcEntry, target, model, boundary, options)
    child = target;
    eta = options.transferBlendMin + rand * (options.transferBlendMax - options.transferBlendMin);

    mixed = (1 - eta) * double(target.rnvec) + eta * double(srcEntry.rnvec);
    mixed = perturb_and_project_rnvec(mixed, model, boundary, options.transferNoise);

    child.rnvec = mixed;
    child = repair_chromosome(child, model, options, false);
end

function rnvec = perturb_and_project_rnvec(rnvec, model, boundary, noiseScale)
    if size(rnvec, 1) > 2
        rangeX = max(1, boundary(1, 1) - boundary(2, 1));
        rangeY = max(1, boundary(1, 2) - boundary(2, 2));
        rangeZ = max(1, boundary(1, 3) - boundary(2, 3));

        rnvec(2:end-1, 1) = rnvec(2:end-1, 1) + noiseScale * rangeX * randn(size(rnvec, 1) - 2, 1);
        rnvec(2:end-1, 2) = rnvec(2:end-1, 2) + noiseScale * rangeY * randn(size(rnvec, 1) - 2, 1);
        rnvec(2:end-1, 3) = rnvec(2:end-1, 3) + 0.5 * noiseScale * rangeZ * randn(size(rnvec, 1) - 2, 1);
    end

    rnvec(:, 1) = max(boundary(2, 1), min(boundary(1, 1), rnvec(:, 1)));
    rnvec(:, 2) = max(boundary(2, 2), min(boundary(1, 2), rnvec(:, 2)));
    rnvec(:, 3) = max(boundary(2, 3), min(boundary(1, 3), rnvec(:, 3)));

    rnvec = sortrows(rnvec, 1);
    rnvec(1, :) = double(model.start);
    rnvec(end, :) = double(model.end);
end

function [piMat, isValid] = sinkhorn_transport(C, a, b, epsilon, maxIter, tol)
    epsilon = max(1e-9, epsilon);
    K = exp(-C / epsilon);
    K = max(K, 1e-300);

    u = ones(size(a));
    v = ones(size(b));
    tiny = 1e-12;

    for it = 1:maxIter
        uPrev = u;
        Kv = K * v + tiny;
        u = a ./ Kv;
        KTu = K' * u + tiny;
        v = b ./ KTu;

        if mod(it, 5) == 0
            if norm(u - uPrev, 1) < tol
                break;
            end
        end
    end

    piMat = (u .* K) .* v';
    mass = sum(piMat(:));
    if ~isfinite(mass) || mass <= 0
        piMat = zeros(size(C));
        isValid = false;
        return;
    end
    piMat = piMat / mass;
    isValid = true;
end

function [srcIdx, tgtIdx] = sample_pairs_from_coupling(piMat, nPairs)
    flat = piMat(:);
    cdf = cumsum(flat);
    totalMass = cdf(end);

    if totalMass <= 0
        m = size(piMat, 1);
        n = size(piMat, 2);
        srcIdx = randi(m, nPairs, 1);
        tgtIdx = randi(n, nPairs, 1);
        return;
    end

    r = rand(nPairs, 1) * totalMass;
    pairIdx = zeros(nPairs, 1);
    for i = 1:nPairs
        pairIdx(i) = find(cdf >= r(i), 1, 'first');
        if isempty(pairIdx(i))
            pairIdx(i) = numel(flat);
        end
    end

    [srcIdx, tgtIdx] = ind2sub(size(piMat), pairIdx);
end

function entries = select_source_entries(archive, currentLambda, currentTheta, options)
    entries = struct('rnvec', {}, 'path', {}, 'objs', {}, 'theta', {}, ...
        'lambda', {}, 'signature', {}, 'age', {}, 'generation', {});

    if isempty(archive.items)
        return;
    end

    items = archive.items;
    lambdas = [items.lambda];
    mask = abs(lambdas - currentLambda) >= options.minLambdaGap;
    if nnz(mask) < options.minSourceEntries
        mask = true(size(lambdas));
    end

    candidates = items(mask);
    if options.feasibleArchiveOnly && ~isempty(candidates)
        feasibleMask = false(1, numel(candidates));
        for i = 1:numel(candidates)
            feasibleMask(i) = all(isfinite(candidates(i).objs));
        end
        if any(feasibleMask)
            candidates = candidates(feasibleMask);
        end
    end
    if isempty(candidates)
        return;
    end

    score = zeros(numel(candidates), 1);
    for i = 1:numel(candidates)
        q = objs_quality(candidates(i).objs, options);
        d = norm(candidates(i).theta - currentTheta) / sqrt(max(1, numel(currentTheta)));
        score(i) = q + options.sourceDistanceBias * d;
    end

    [~, ord] = sort(score, 'ascend');
    k = min(options.sourcePoolSize, numel(candidates));
    entries = candidates(ord(1:k));
end

function history = initialize_history(generations, options, beta0)
    history = struct();
    history.scheduler = options.scheduler;
    history.generation = zeros(generations, 1);
    history.lambda = nan(generations, 1);
    history.hypervolume = nan(generations, 1);
    history.diversity = nan(generations, 1);
    history.feasibility = nan(generations, 1);
    history.beta = nan(generations, 1);
    history.beta(1) = beta0;
    history.tau = nan(generations, 1);
    history.tmtm = nan(generations, 1);
    history.nti = nan(generations, 1);
    history.dM = nan(generations, 1);
    history.transferGain = nan(generations, 1);
    history.controlGain = nan(generations, 1);
    history.entropy = nan(generations, 1);
    history.transferPairs = zeros(generations, 1);
    history.transferUsed = false(generations, 1);
end

function state = build_scheduler_state(population, lambda, history, gen, generations)
    popObj = population_to_obj(population, numel(population(1).objs));

    state = struct();
    state.lambda = lambda;
    state.diversity = population_diversity(popObj);
    state.feasibility = mean(all(isfinite(popObj), 2));
    state.progress = gen / max(1, generations);

    if gen > 1
        prevHV = history.hypervolume(1:gen-1);
        prevHV = prevHV(isfinite(prevHV));
        if numel(prevHV) >= 2
            state.improving = prevHV(end) > prevHV(end-1);
        else
            state.improving = false;
        end
    else
        state.improving = false;
    end
end

function lambdaNew = schedule_lambda(lambdaCurrent, state, options)
    scheduler = lower(strtrim(options.scheduler));
    switch scheduler
        case 'gradual'
            lambdaNew = lambdaCurrent + options.stepSize;
        otherwise
            % Lightweight feasibility-aware policy.
            if state.feasibility > options.feasHigh && state.diversity > options.divThreshold
                lambdaNew = lambdaCurrent + options.stepSize;
            elseif state.feasibility < options.feasLow
                lambdaNew = lambdaCurrent - options.stepSizeDown;
            else
                drift = options.progressDrift * (0.5 + state.progress);
                if state.improving
                    lambdaNew = lambdaCurrent + drift;
                else
                    lambdaNew = lambdaCurrent;
                end
            end
    end
    if options.enforceProgressFloor && state.progress >= options.lambdaFloorStart && options.lambdaFloorStart < 1
        floorByProgress = (state.progress - options.lambdaFloorStart) / (1 - options.lambdaFloorStart);
        lambdaNew = max(lambdaNew, floorByProgress);
    end
    lambdaNew = clip01(lambdaNew);
end

function beta = update_beta(beta, tau, options)
    if ~isfinite(tau)
        tau = 0;
    end
    beta = beta + options.betaEta * sign(-tau);
    beta = max(options.betaMin, min(options.betaMax, beta));
end

function archive = initialize_archive(options)
    archive = struct();
    archive.items = struct('rnvec', {}, 'path', {}, 'objs', {}, 'theta', {}, ...
        'lambda', {}, 'signature', {}, 'age', {}, 'generation', {});
    archive.maxSize = options.archiveMaxSize;
    archive.counter = 0;
end

function archive = append_archive_from_population(archive, population, model, theta, ...
    lambda, options, frontNo, generation)

    if isempty(population)
        return;
    end

    if nargin < 7 || isempty(frontNo)
        frontNo = ones(numel(population), 1);
    end
    if nargin < 8
        generation = 0;
    end

    bestFront = min(frontNo);
    candidateIdx = find(frontNo == bestFront);
    if isempty(candidateIdx)
        candidateIdx = 1:numel(population);
    end

    if options.feasibleArchiveOnly
        feasibleMask = false(size(candidateIdx));
        for i = 1:numel(candidateIdx)
            feasibleMask(i) = all(isfinite(population(candidateIdx(i)).objs));
        end
        if any(feasibleMask)
            candidateIdx = candidateIdx(feasibleMask);
        end
    end
    if isempty(candidateIdx)
        candidateIdx = 1:numel(population);
    end

    q = individual_quality_scores(population(candidateIdx), options);
    [~, ord] = sort(q, 'ascend');
    k = min(options.archiveInjectCount, numel(candidateIdx));
    chosen = candidateIdx(ord(1:k));

    for i = 1:numel(chosen)
        idx = chosen(i);
        entry = struct();
        entry.rnvec = population(idx).rnvec;
        entry.path = population(idx).path;
        entry.objs = population(idx).objs;
        entry.theta = theta;
        entry.lambda = lambda;
        entry.signature = topology_signature(entry.path, model, options);
        archive.counter = archive.counter + 1;
        entry.age = archive.counter;
        entry.generation = generation;
        archive.items(end+1) = entry; %#ok<AGROW>
    end

    if numel(archive.items) > archive.maxSize
        [~, ordAge] = sort([archive.items.age], 'descend');
        keep = ordAge(1:archive.maxSize);
        archive.items = archive.items(keep);
    end
end

function sig = topology_signature(path, model, options)
    sig = zeros(1, signature_dimension(options));
    if isempty(path) || size(path, 2) < 2 || size(path, 1) < 2
        return;
    end

    xy = double(path(:, 1:2));
    dx = max(1, double(model.xmax) - double(model.xmin));
    dy = max(1, double(model.ymax) - double(model.ymin));
    mapDiag = sqrt(dx^2 + dy^2);

    dxy = diff(xy, 1, 1);
    segLen = sqrt(sum(dxy.^2, 2));
    pathLenNorm = sum(segLen) / mapDiag;

    heading = atan2(dxy(:, 2), dxy(:, 1));
    if numel(heading) >= 2
        turn = wrap_to_pi(diff(heading));
        meanTurn = mean(abs(turn)) / pi;
        signedTurn = sum(turn) / (pi * max(1, numel(turn)));
        turnStd = std(turn) / pi;
    else
        meanTurn = 0;
        signedTurn = 0;
        turnStd = 0;
    end

    sig(1:4) = [pathLenNorm, meanTurn, signedTurn, turnStd];

    obsFeat = zeros(1, options.signatureMaxObstacles * 3);
    [centers, radii] = extract_obstacles(model, options.signatureMaxObstacles);
    if ~isempty(centers)
        baseDir = xy(end, :) - xy(1, :);
        if norm(baseDir) < 1e-12
            baseDir = [1, 0];
        end
        for k = 1:size(centers, 1)
            c = centers(k, :);
            r = radii(k);
            dist = sqrt((xy(:, 1) - c(1)).^2 + (xy(:, 2) - c(2)).^2);
            [minDist, idx] = min(dist);
            sideVec = xy(idx, :) - c;
            side = sign(baseDir(1) * sideVec(2) - baseDir(2) * sideVec(1));
            if ~isfinite(side)
                side = 0;
            end

            ang = unwrap(atan2(xy(:, 2) - c(2), xy(:, 1) - c(1)));
            winding = (ang(end) - ang(1)) / (2 * pi);
            clearance = (minDist - r) / mapDiag;

            base = 3 * (k - 1);
            obsFeat(base + 1:base + 3) = [side, winding, clearance];
        end
    end
    sig(5:end) = obsFeat;
end

function d = signature_distance(sigA, sigB)
    if isempty(sigA) || isempty(sigB)
        d = 1;
        return;
    end
    delta = sigA - sigB;
    d = norm(delta) / sqrt(max(1, numel(delta)));
    d = min(1, max(0, d));
end

function n = signature_dimension(options)
    n = 4 + 3 * options.signatureMaxObstacles;
end

function [centers, radii] = extract_obstacles(model, maxObs)
    centers = [];
    radii = [];

    if isfield(model, 'nofly_c') && isfield(model, 'nofly_r') && ~isempty(model.nofly_c)
        c = double(model.nofly_c);
        if isvector(c) && numel(c) == 2
            c = reshape(c, 1, 2);
        end
        if size(c, 2) > 2
            c = c(:, 1:2);
        end

        r = double(model.nofly_r);
        if isempty(r)
            r = 0;
        end
        if isscalar(r)
            r = repmat(r, size(c, 1), 1);
        else
            r = r(:);
            if numel(r) < size(c, 1)
                r(end+1:size(c, 1)) = r(end);
            end
        end

        centers = [centers; c];
        radii = [radii; r(1:size(c, 1))];
    end

    if isfield(model, 'threats') && ~isempty(model.threats)
        th = double(model.threats);
        if size(th, 2) >= 4
            centers = [centers; th(:, 1:2)];
            radii = [radii; th(:, 4)];
        end
    end

    if isempty(centers)
        return;
    end

    valid = all(isfinite(centers), 2) & isfinite(radii) & radii > 0;
    centers = centers(valid, :);
    radii = radii(valid);

    if isempty(centers)
        return;
    end

    [~, ord] = sort(radii, 'descend');
    k = min(maxObs, numel(ord));
    ord = ord(1:k);
    centers = centers(ord, :);
    radii = radii(ord);
end

function theta = extract_theta_vector(model, modelHard, options)
    safeDistCur = get_model_scalar(model, 'safeDist', options.safeDistDefault);
    safeDistHard = get_model_scalar(modelHard, 'safeDist', max(1, safeDistCur));

    safeHCur = get_model_scalar(model, 'safeH', options.safeHDefault);
    safeHHard = get_model_scalar(modelHard, 'safeH', max(1, safeHCur));

    noflyRatio = obstacle_area_ratio(model);

    roughCur = terrain_roughness(model);
    roughHard = terrain_roughness(modelHard);
    if roughHard <= 0
        roughHard = 1;
    end

    thetaCur = get_model_scalar(model, 'theta', 0);
    thetaHard = get_model_scalar(modelHard, 'theta', max(1, abs(thetaCur)));

    theta = [ ...
        safeDistCur / max(1e-12, safeDistHard), ...
        safeHCur / max(1e-12, safeHHard), ...
        noflyRatio, ...
        roughCur / max(1e-12, roughHard), ...
        thetaCur / max(1, abs(thetaHard)) ...
    ];
    theta(~isfinite(theta)) = 0;
end

function ratio = obstacle_area_ratio(model)
    [~, radii] = extract_obstacles(model, inf);
    if isempty(radii)
        ratio = 0;
        return;
    end
    areaObs = sum(pi * (double(radii(:)).^2));
    areaMap = max(1, (double(model.xmax) - double(model.xmin)) * (double(model.ymax) - double(model.ymin)));
    ratio = min(1, areaObs / areaMap);
end

function r = terrain_roughness(model)
    if ~isfield(model, 'H') || isempty(model.H)
        r = 0;
        return;
    end
    H = double(model.H);
    [gx, gy] = gradient(H);
    r = mean(abs(gx(:))) + mean(abs(gy(:)));
    if ~isfinite(r)
        r = 0;
    end
end

function value = get_model_scalar(model, fieldName, defaultValue)
    value = defaultValue;
    if isfield(model, fieldName) && ~isempty(model.(fieldName))
        v = model.(fieldName);
        if isnumeric(v) && isfinite(v(1))
            value = double(v(1));
        end
    end
end

function scores = individual_quality_scores(individuals, options)
    scores = zeros(numel(individuals), 1);
    for i = 1:numel(individuals)
        scores(i) = chromosome_quality(individuals(i), options);
    end
end

function score = chromosome_quality(chrom, options)
    score = objs_quality(chrom.objs, options) + options.constraintPriority * individual_constraint_value(chrom);
end

function score = objs_quality(objs, options)
    objs = double(objs(:)');
    if isempty(objs) || any(~isfinite(objs))
        score = 1e9;
        return;
    end
    w = options.objWeights;
    if numel(w) ~= numel(objs)
        w = ones(1, numel(objs));
    end
    score = sum(w(:)' .* objs);
end

function cv = individual_constraint_value(individual)
    cv = 0;
    if isfield(individual, 'cons') && ~isempty(individual.cons)
        c = double(individual.cons(1));
        if isfinite(c)
            cv = max(0, c);
            return;
        end
    end
    if isfield(individual, 'objs') && ~isempty(individual.objs) && any(~isfinite(individual.objs))
        cv = 1e6;
    end
end

function info = default_ot_info(beta)
    info = struct();
    info.used = false;
    info.tau = 0;
    info.tmtm = NaN;
    info.nti = NaN;
    info.dM = NaN;
    info.gainTransfer = NaN;
    info.gainNoTransfer = NaN;
    info.betaBefore = beta;
    info.betaAfter = beta;
    info.entropy = NaN;
    info.pairCount = 0;
end

function options = parse_tcot_options(params)
    options = struct();

    % Scheduler.
    options.scheduler = 'performance';
    options.lambda0 = 0.0;
    options.stepSize = 0.08;
    options.stepSizeDown = 0.06;
    options.progressDrift = 0.01;
    options.feasHigh = 0.80;
    options.feasLow = 0.35;
    options.divThreshold = 0.08;

    % Easy task shaping.
    options.easySafeDistScale = 0.35;
    options.easySafeHScale = 0.65;
    options.easyNoFlyScale = 0.40;
    options.easyThreatScale = 0.40;
    options.easyTerrainScale = 0.80;

    % Logging and metrics.
    options.logInterval = 20;
    options.hvSamplesLog = 800;

    % OT transfer core.
    options.otInterval = 5;
    options.transferFraction = 0.50;
    options.sourcePoolSize = 120;
    options.targetPoolSize = 120;
    options.minArchiveForOT = 30;
    options.archiveMaxSize = 500;
    options.archiveInjectCount = 16;
    options.minLambdaGap = 0.05;
    options.minSourceEntries = 8;
    options.sourceDistanceBias = 0.25;

    options.alpha = 1.0;
    options.beta0 = 1.2;
    options.betaMin = 0.1;
    options.betaMax = 8.0;
    options.betaEta = 0.10;
    options.gamma = 0.5;
    options.otEpsilon = 0.08;
    options.otMaxIter = 80;
    options.otTol = 1e-4;

    options.useCounterfactual = true;
    options.counterfactualControlCount = 16;

    options.transferBlendMin = 0.25;
    options.transferBlendMax = 0.75;
    options.transferNoise = 0.015;

    options.signatureMaxObstacles = 4;
    options.ntiTolerance = 1e-9;
    options.objWeights = [1, 1, 1, 1];

    options.safeDistDefault = 20;
    options.safeHDefault = 20;

    % Hard-target stabilization and rescue.
    options.enforceProgressFloor = true;
    options.lambdaFloorStart = 0.70;
    options.finalRepairPasses = 4;
    options.rescueEnabled = true;
    options.rescueBatchFactor = 3;
    options.rescueMaxAttempts = 4;
    options.rescueTargetFeasibility = 0.20;

    % Domain-specific feasibility controls.
    options.feasibilityPhaseRatio = 0.35;
    options.feasibilityLambdaCap = 0.55;
    options.enableOTInFeasibilityPhase = false;
    options.domainGuidedFraction = 0.20;
    options.domainGuidedFractionFeas = 0.35;
    options.domainGuidedBlend = 0.45;
    options.domainGuidedNoise = 0.01;
    options.domainGuidedUseArchive = true;
    options.domainMutationTrials = 3;
    options.feasibleArchiveOnly = true;
    options.constraintPriority = 100;
    options.feasibilityNoiseBoost = 1.8;
    options.obstacleNudgeIters = 3;
    options.clearanceLiftIters = 3;
    options.minClearanceSafeDistScale = 0.30;
    options.baseClearanceMargin = 0.5;
    options.aggressiveClearanceMargin = 0.8;
    options.feasibilityClearanceBoost = 1.2;
    options.nonfiniteViolationBoost = 5.0;

    if isfield(params, 'ctm') && isstruct(params.ctm)
        ctm = params.ctm;
        options = override_if_present(options, ctm, 'scheduler');
        options = override_if_present(options, ctm, 'lambda0');
        options = override_if_present(options, ctm, 'stepSize');
        options = override_if_present(options, ctm, 'stepSizeDown');
        options = override_if_present(options, ctm, 'progressDrift');
        options = override_if_present(options, ctm, 'feasHigh');
        options = override_if_present(options, ctm, 'feasLow');
        options = override_if_present(options, ctm, 'divThreshold');

        options = override_if_present(options, ctm, 'easySafeDistScale');
        options = override_if_present(options, ctm, 'easySafeHScale');
        options = override_if_present(options, ctm, 'easyNoFlyScale');
        options = override_if_present(options, ctm, 'easyThreatScale');
        options = override_if_present(options, ctm, 'easyTerrainScale');

        options = override_if_present(options, ctm, 'logInterval');
        options = override_if_present(options, ctm, 'hvSamplesLog');

        options = override_if_present(options, ctm, 'otInterval');
        options = override_if_present(options, ctm, 'transferFraction');
        options = override_if_present(options, ctm, 'sourcePoolSize');
        options = override_if_present(options, ctm, 'targetPoolSize');
        options = override_if_present(options, ctm, 'minArchiveForOT');
        options = override_if_present(options, ctm, 'archiveMaxSize');
        options = override_if_present(options, ctm, 'archiveInjectCount');
        options = override_if_present(options, ctm, 'minLambdaGap');
        options = override_if_present(options, ctm, 'minSourceEntries');
        options = override_if_present(options, ctm, 'sourceDistanceBias');

        options = override_if_present(options, ctm, 'alpha');
        options = override_if_present(options, ctm, 'beta0');
        options = override_if_present(options, ctm, 'betaMin');
        options = override_if_present(options, ctm, 'betaMax');
        options = override_if_present(options, ctm, 'betaEta');
        options = override_if_present(options, ctm, 'gamma');
        options = override_if_present(options, ctm, 'otEpsilon');
        options = override_if_present(options, ctm, 'otMaxIter');
        options = override_if_present(options, ctm, 'otTol');

        options = override_if_present(options, ctm, 'useCounterfactual');
        options = override_if_present(options, ctm, 'counterfactualControlCount');

        options = override_if_present(options, ctm, 'transferBlendMin');
        options = override_if_present(options, ctm, 'transferBlendMax');
        options = override_if_present(options, ctm, 'transferNoise');

        options = override_if_present(options, ctm, 'signatureMaxObstacles');
        options = override_if_present(options, ctm, 'ntiTolerance');
        options = override_if_present(options, ctm, 'objWeights');

        options = override_if_present(options, ctm, 'enforceProgressFloor');
        options = override_if_present(options, ctm, 'lambdaFloorStart');
        options = override_if_present(options, ctm, 'finalRepairPasses');
        options = override_if_present(options, ctm, 'rescueEnabled');
        options = override_if_present(options, ctm, 'rescueBatchFactor');
        options = override_if_present(options, ctm, 'rescueMaxAttempts');
        options = override_if_present(options, ctm, 'rescueTargetFeasibility');

        options = override_if_present(options, ctm, 'feasibilityPhaseRatio');
        options = override_if_present(options, ctm, 'feasibilityLambdaCap');
        options = override_if_present(options, ctm, 'enableOTInFeasibilityPhase');
        options = override_if_present(options, ctm, 'domainGuidedFraction');
        options = override_if_present(options, ctm, 'domainGuidedFractionFeas');
        options = override_if_present(options, ctm, 'domainGuidedBlend');
        options = override_if_present(options, ctm, 'domainGuidedNoise');
        options = override_if_present(options, ctm, 'domainGuidedUseArchive');
        options = override_if_present(options, ctm, 'domainMutationTrials');
        options = override_if_present(options, ctm, 'feasibleArchiveOnly');
        options = override_if_present(options, ctm, 'constraintPriority');
        options = override_if_present(options, ctm, 'feasibilityNoiseBoost');
        options = override_if_present(options, ctm, 'obstacleNudgeIters');
        options = override_if_present(options, ctm, 'clearanceLiftIters');
        options = override_if_present(options, ctm, 'minClearanceSafeDistScale');
        options = override_if_present(options, ctm, 'baseClearanceMargin');
        options = override_if_present(options, ctm, 'aggressiveClearanceMargin');
        options = override_if_present(options, ctm, 'feasibilityClearanceBoost');
        options = override_if_present(options, ctm, 'nonfiniteViolationBoost');
    end

    options.lambda0 = clip01(options.lambda0);
    options.stepSize = max(0, min(1, options.stepSize));
    options.stepSizeDown = max(0, min(1, options.stepSizeDown));
    options.progressDrift = max(0, min(1, options.progressDrift));
    options.feasHigh = max(0, min(1, options.feasHigh));
    options.feasLow = max(0, min(1, options.feasLow));
    options.divThreshold = max(0, options.divThreshold);

    options.logInterval = max(1, round(options.logInterval));
    options.hvSamplesLog = max(100, round(options.hvSamplesLog));

    options.otInterval = max(1, round(options.otInterval));
    options.transferFraction = max(0, min(1, options.transferFraction));
    options.sourcePoolSize = max(4, round(options.sourcePoolSize));
    options.targetPoolSize = max(4, round(options.targetPoolSize));
    options.minArchiveForOT = max(2, round(options.minArchiveForOT));
    options.archiveMaxSize = max(32, round(options.archiveMaxSize));
    options.archiveInjectCount = max(1, round(options.archiveInjectCount));
    options.minLambdaGap = max(0, min(1, options.minLambdaGap));
    options.minSourceEntries = max(1, round(options.minSourceEntries));
    options.sourceDistanceBias = max(0, options.sourceDistanceBias);

    options.alpha = max(0, options.alpha);
    options.beta0 = max(1e-6, options.beta0);
    options.betaMin = max(1e-6, options.betaMin);
    options.betaMax = max(options.betaMin, options.betaMax);
    options.betaEta = max(0, options.betaEta);
    options.gamma = max(0, options.gamma);
    options.otEpsilon = max(1e-6, options.otEpsilon);
    options.otMaxIter = max(5, round(options.otMaxIter));
    options.otTol = max(1e-8, options.otTol);

    options.useCounterfactual = logical(options.useCounterfactual);
    options.counterfactualControlCount = max(0, round(options.counterfactualControlCount));

    options.transferBlendMin = max(0, min(1, options.transferBlendMin));
    options.transferBlendMax = max(options.transferBlendMin, min(1, options.transferBlendMax));
    options.transferNoise = max(0, options.transferNoise);

    options.signatureMaxObstacles = max(0, round(options.signatureMaxObstacles));
    options.ntiTolerance = max(0, options.ntiTolerance);

    if isempty(options.objWeights)
        options.objWeights = [1, 1, 1, 1];
    end
    options.objWeights = double(options.objWeights(:)');

    options.enforceProgressFloor = logical(options.enforceProgressFloor);
    options.lambdaFloorStart = clip01(options.lambdaFloorStart);
    options.finalRepairPasses = max(1, round(options.finalRepairPasses));
    options.rescueEnabled = logical(options.rescueEnabled);
    options.rescueBatchFactor = max(1, round(options.rescueBatchFactor));
    options.rescueMaxAttempts = max(1, round(options.rescueMaxAttempts));
    options.rescueTargetFeasibility = max(0, min(1, options.rescueTargetFeasibility));

    options.feasibilityPhaseRatio = clip01(options.feasibilityPhaseRatio);
    options.feasibilityLambdaCap = clip01(options.feasibilityLambdaCap);
    options.enableOTInFeasibilityPhase = logical(options.enableOTInFeasibilityPhase);
    options.domainGuidedFraction = max(0, min(1, options.domainGuidedFraction));
    options.domainGuidedFractionFeas = max(0, min(1, options.domainGuidedFractionFeas));
    options.domainGuidedBlend = max(0, min(1, options.domainGuidedBlend));
    options.domainGuidedNoise = max(0, options.domainGuidedNoise);
    options.domainGuidedUseArchive = logical(options.domainGuidedUseArchive);
    options.domainMutationTrials = max(1, round(options.domainMutationTrials));
    options.feasibleArchiveOnly = logical(options.feasibleArchiveOnly);
    options.constraintPriority = max(0, options.constraintPriority);
    options.feasibilityNoiseBoost = max(1, options.feasibilityNoiseBoost);
    options.obstacleNudgeIters = max(1, round(options.obstacleNudgeIters));
    options.clearanceLiftIters = max(1, round(options.clearanceLiftIters));
    options.minClearanceSafeDistScale = max(0, options.minClearanceSafeDistScale);
    options.baseClearanceMargin = max(0, options.baseClearanceMargin);
    options.aggressiveClearanceMargin = max(0, options.aggressiveClearanceMargin);
    options.feasibilityClearanceBoost = max(1, options.feasibilityClearanceBoost);
    options.nonfiniteViolationBoost = max(0, options.nonfiniteViolationBoost);
end

function options = override_if_present(options, src, fieldName)
    if isfield(src, fieldName) && ~isempty(src.(fieldName))
        options.(fieldName) = src.(fieldName);
    end
end

function modelEasy = build_easy_model(modelHard, options)
    modelEasy = modelHard;

    if isfield(modelHard, 'safeDist') && ~isempty(modelHard.safeDist)
        modelEasy.safeDist = max(1, options.easySafeDistScale * double(modelHard.safeDist));
    end

    if isfield(modelHard, 'safeH') && ~isempty(modelHard.safeH)
        modelEasy.safeH = max(1, options.easySafeHScale * double(modelHard.safeH));
    end

    if isfield(modelHard, 'nofly_r') && ~isempty(modelHard.nofly_r)
        modelEasy.nofly_r = options.easyNoFlyScale * double(modelHard.nofly_r);
    end

    if isfield(modelHard, 'threats') && ~isempty(modelHard.threats)
        modelEasy.threats = modelHard.threats;
        if size(modelEasy.threats, 2) >= 4
            modelEasy.threats(:, 4) = options.easyThreatScale * double(modelHard.threats(:, 4));
        end
    end

    if isfield(modelHard, 'H') && ~isempty(modelHard.H)
        h = double(modelHard.H);
        hMin = min(h(:));
        modelEasy.H = hMin + options.easyTerrainScale * (h - hMin);
    end
end

function model = interpolate_models(modelEasy, modelHard, lambda)
    lambda = clip01(lambda);
    model = modelHard;

    if isfield(modelHard, 'safeDist') && isfield(modelEasy, 'safeDist')
        model.safeDist = (1 - lambda) * double(modelEasy.safeDist) + lambda * double(modelHard.safeDist);
    end

    if isfield(modelHard, 'safeH') && isfield(modelEasy, 'safeH')
        model.safeH = (1 - lambda) * double(modelEasy.safeH) + lambda * double(modelHard.safeH);
    end

    if isfield(modelHard, 'nofly_r') && isfield(modelEasy, 'nofly_r') && ...
            numel(modelHard.nofly_r) == numel(modelEasy.nofly_r)
        model.nofly_r = (1 - lambda) * double(modelEasy.nofly_r) + lambda * double(modelHard.nofly_r);
    end

    if isfield(modelHard, 'threats') && isfield(modelEasy, 'threats') && ...
            isequal(size(modelHard.threats), size(modelEasy.threats))
        model.threats = modelHard.threats;
        if size(model.threats, 2) >= 4
            model.threats(:, 4) = (1 - lambda) * double(modelEasy.threats(:, 4)) + ...
                                  lambda * double(modelHard.threats(:, 4));
        end
    end

    if isfield(modelHard, 'H') && isfield(modelEasy, 'H') && isequal(size(modelHard.H), size(modelEasy.H))
        model.H = (1 - lambda) * double(modelEasy.H) + lambda * double(modelHard.H);
    end
end

function population = finalize_on_hard_model(population, modelHard, pop, M, options, boundaryHard)
    if isempty(population)
        return;
    end

    [population, frontNo, crowdDis] = EnvironmentalSelection(population, pop, numel(population), M);
    for pass = 1:options.finalRepairPasses
        population = reevaluate_population(population, modelHard, options, true);
        [population, frontNo, crowdDis] = EnvironmentalSelection(population, pop, numel(population), M);
        popObj = population_to_obj(population, M);
        finiteRatio = mean(all(isfinite(popObj), 2));
        if finiteRatio >= options.rescueTargetFeasibility
            break;
        end

        matingPool = TournamentSelection(2, pop, frontNo, -crowdDis);
        offspring = F_operator(population, matingPool', boundaryHard, modelHard);
        offspring = repair_population(offspring, modelHard, options, true);
        merged = [population, offspring];
        [population, frontNo, crowdDis] = EnvironmentalSelection(merged, pop, numel(merged), M);
    end

    popObj = population_to_obj(population, M);
    finiteRatio = mean(all(isfinite(popObj), 2));
    if options.rescueEnabled && finiteRatio < options.rescueTargetFeasibility
        population = rescue_population(population, modelHard, pop, M, options);
    end
end

function population = rescue_population(population, modelHard, pop, M, options)
    survivors = population;
    for attempt = 1:options.rescueMaxAttempts
        nNew = max(pop, options.rescueBatchFactor * pop);
        newPop = Chromosome.empty(0, nNew);
        for i = 1:nNew
            c = Chromosome(modelHard);
            if mod(i, 2) == 0
                c = seed_corridor_candidate(c, modelHard, options, i);
            else
                c = initialize(c, modelHard);
            end
            c = repair_chromosome(c, modelHard, options, true);
            newPop(i) = c;
        end

        pool = [survivors, newPop];
        [survivors, ~, ~] = EnvironmentalSelection(pool, pop, numel(pool), M);

        popObj = population_to_obj(survivors, M);
        if mean(all(isfinite(popObj), 2)) >= options.rescueTargetFeasibility
            break;
        end
        if any(all(isfinite(popObj), 2))
            break;
        end
        if attempt < options.rescueMaxAttempts
            survivors = reevaluate_population(survivors, modelHard, options, true);
        end
    end
    population = survivors;
end

function population = repair_population(population, model, options, aggressive)
    if nargin < 4
        aggressive = false;
    end
    for i = 1:numel(population)
        population(i) = repair_chromosome(population(i), model, options, aggressive);
    end
end

function chrom = repair_chromosome(chrom, model, options, aggressive)
    if nargin < 3 || isempty(options) || ~isstruct(options)
        options = struct();
    end
    if nargin < 4
        aggressive = false;
    end
    if ~isfield(options, 'signatureMaxObstacles') || isempty(options.signatureMaxObstacles)
        options.signatureMaxObstacles = 4;
    end
    if ~isfield(options, 'safeDistDefault') || isempty(options.safeDistDefault)
        options.safeDistDefault = 20;
    end

    chrom.path = chrom.rnvec;
    chrom = adjust_constraint_turning_angle(chrom, model);
    chrom.path = nudge_path_from_obstacles(chrom.path, model, options, aggressive);
    chrom = adjust_constraint_turning_angle(chrom, model);
    chrom.path = lift_path_clearance(chrom.path, model, options, aggressive);
    chrom.rnvec = chrom.path;
    chrom = evaluate(chrom, model);
    chrom.cons = path_constraint_violation(chrom.path, chrom.objs, model, options);
    if ~isfinite(chrom.cons)
        chrom.cons = 1e9;
    end
end

function path = nudge_path_from_obstacles(path, model, options, aggressive)
    if size(path, 1) <= 2
        return;
    end

    [centers, radii] = extract_obstacles(model, max(options.signatureMaxObstacles, 16));
    if isempty(centers)
        return;
    end

    droneSize = get_model_scalar(model, 'droneSize', 1);
    if ~isfield(model, 'droneSize') && isfield(model, 'drone_size')
        droneSize = get_model_scalar(model, 'drone_size', droneSize);
    end
    safeDist = get_model_scalar(model, 'safeDist', options.safeDistDefault);
    if ~isfield(model, 'safeDist') && isfield(model, 'safe_dist')
        safeDist = get_model_scalar(model, 'safe_dist', safeDist);
    end
    minClear = max(1, droneSize + options.minClearanceSafeDistScale * safeDist);
    if aggressive
        minClear = minClear * options.feasibilityClearanceBoost;
    end

    nIters = options.obstacleNudgeIters;
    if aggressive
        nIters = nIters + 2;
    end

    for iter = 1:nIters
        for i = 2:size(path, 1)-1
            p = path(i, 1:2);
            for k = 1:size(centers, 1)
                c = centers(k, :);
                r = radii(k);
                req = r + minClear;
                d = norm(p - c);
                if d < req
                    if d < 1e-9
                        dir = randn(1, 2);
                        dir = dir / max(norm(dir), 1e-12);
                    else
                        dir = (p - c) / d;
                    end
                    p = c + req * dir;
                end
            end
            p(1) = max(model.xmin, min(model.xmax, p(1)));
            p(2) = max(model.ymin, min(model.ymax, p(2)));
            path(i, 1:2) = p;
        end

        for j = 1:size(path, 1)-1
            p1 = path(j, 1:2);
            p2 = path(j+1, 1:2);
            mid = 0.5 * (p1 + p2);
            for k = 1:size(centers, 1)
                c = centers(k, :);
                r = radii(k);
                req = r + minClear;
                dSeg = dist_point_to_segment_2d_local(c, p1, p2);
                if dSeg < req
                    dir = mid - c;
                    dn = norm(dir);
                    if dn < 1e-9
                        dir = randn(1, 2);
                        dn = norm(dir);
                    end
                    dir = dir / max(dn, 1e-12);
                    shift = (req - dSeg) * dir;
                    if j > 1
                        path(j, 1:2) = path(j, 1:2) + 0.5 * shift;
                    end
                    if j+1 < size(path, 1)
                        path(j+1, 1:2) = path(j+1, 1:2) + 0.5 * shift;
                    end
                end
            end
        end

        path(:, 1) = max(model.xmin, min(model.xmax, path(:, 1)));
        path(:, 2) = max(model.ymin, min(model.ymax, path(:, 2)));
    end

    path = sortrows(path, 1);
    path(1, 1:2) = double(model.start(1:2));
    path(end, 1:2) = double(model.end(1:2));
end

function path = lift_path_clearance(path, model, options, aggressive)
    if size(path, 1) < 2 || ~isfield(model, 'H') || isempty(model.H)
        return;
    end

    stepSize = 1;
    if isfield(model, 'collisionStep') && isnumeric(model.collisionStep) && isfinite(model.collisionStep)
        stepSize = max(1e-3, double(model.collisionStep));
    end

    droneSize = get_model_scalar(model, 'droneSize', 1);
    if ~isfield(model, 'droneSize') && isfield(model, 'drone_size')
        droneSize = get_model_scalar(model, 'drone_size', droneSize);
    end
    targetRel = droneSize + options.baseClearanceMargin;
    if aggressive
        targetRel = targetRel + options.aggressiveClearanceMargin;
    end

    zminRel = get_model_scalar(model, 'zmin', 0) + 1e-3;
    zmaxRel = get_model_scalar(model, 'zmax', 300) - 1e-3;
    zmaxRel = max(zmaxRel, zminRel + 1e-3);

    nIters = options.clearanceLiftIters;
    if aggressive
        nIters = nIters + 2;
    end

    for iter = 1:nIters
        dense = interpolate_path_local(path, stepSize);
        xi = max(1, min(model.xmax, round(dense(:, 1))));
        yi = max(1, min(model.ymax, round(dense(:, 2))));
        g = model.H(sub2ind(size(model.H), yi, xi));
        minRel = min(dense(:, 3) - g);
        if minRel >= targetRel
            break;
        end
        delta = targetRel - minRel;
        path(2:end-1, 3) = path(2:end-1, 3) + delta;

        xi2 = max(1, min(model.xmax, round(path(:, 1))));
        yi2 = max(1, min(model.ymax, round(path(:, 2))));
        g2 = model.H(sub2ind(size(model.H), yi2, xi2));
        zRel = path(:, 3) - g2;
        zRel = min(max(zRel, zminRel), zmaxRel);
        path(:, 3) = g2 + zRel;
        path(1, :) = double(model.start);
        path(end, :) = double(model.end);
    end
end

function dense = interpolate_path_local(path, stepSize)
    nPoints = size(path, 1);
    nSegments = nPoints - 1;
    stepsPerSegment = zeros(nSegments, 1);
    for i = 1:nSegments
        dist = norm(path(i+1, :) - path(i, :));
        stepsPerSegment(i) = max(1, ceil(dist / stepSize));
    end

    totalPoints = 1 + sum(stepsPerSegment);
    dense = zeros(totalPoints, 3);
    idx = 1;
    dense(idx, :) = path(1, :);
    for i = 1:nSegments
        p1 = path(i, :);
        p2 = path(i+1, :);
        steps = stepsPerSegment(i);
        for s = 1:steps
            t = s / steps;
            idx = idx + 1;
            dense(idx, :) = (1 - t) * p1 + t * p2;
        end
    end
end

function d = dist_point_to_segment_2d_local(p, a, b)
    ab = b - a;
    ap = p - a;
    ab2 = dot(ab, ab);
    if ab2 <= 1e-12
        d = norm(ap);
        return;
    end
    t = dot(ap, ab) / ab2;
    t = max(0, min(1, t));
    proj = a + t * ab;
    d = norm(p - proj);
end

function chrom = seed_corridor_candidate(chrom, model, options, seedIdx)
    nControl = size(chrom.rnvec, 1);
    t = linspace(0, 1, nControl)';

    startXY = reshape(double(model.start(1:2)), 1, []);
    endXY = reshape(double(model.end(1:2)), 1, []);
    baseXY = startXY + t * (endXY - startXY);

    dirVec = endXY - startXY;
    dn = norm(dirVec);
    if dn < 1e-12
        perp = [0, 1];
    else
        dirVec = dirVec / dn;
        perp = [-dirVec(2), dirVec(1)];
    end

    span = max(1, norm([double(model.xmax - model.xmin), double(model.ymax - model.ymin)]));
    amp = 0.05 * span * (1 + 0.3 * sin(seedIdx));
    phase = 2 * pi * mod(seedIdx, 11) / 11;
    lateral = amp * sin(2 * pi * t + phase) .* (t .* (1 - t));
    xy = baseXY + lateral * perp;

    xy(:, 1) = max(model.xmin, min(model.xmax, xy(:, 1)));
    xy(:, 2) = max(model.ymin, min(model.ymax, xy(:, 2)));

    if ~isfield(model, 'H') || isempty(model.H)
        chrom = initialize(chrom, model);
        return;
    end

    safeH = get_model_scalar(model, 'safeH', options.safeHDefault);
    zRelTarget = max(get_model_scalar(model, 'zmin', 0) + safeH, safeH + options.baseClearanceMargin);
    z = zeros(nControl, 1);
    for i = 1:nControl
        xi = max(1, min(size(model.H, 2), round(xy(i, 1))));
        yi = max(1, min(size(model.H, 1), round(xy(i, 2))));
        z(i) = double(model.H(yi, xi)) + zRelTarget;
    end

    path = [xy, z];
    path = sortrows(path, 1);
    path(1, :) = double(model.start);
    path(end, :) = double(model.end);

    chrom.path = path;
    chrom.rnvec = path;
end

function cv = path_constraint_violation(path, objs, model, options)
    cv = 0;
    if isempty(path) || size(path, 2) ~= 3 || size(path, 1) < 2
        cv = 1e9;
        return;
    end

    x = double(path(:, 1));
    y = double(path(:, 2));
    zAbs = double(path(:, 3));

    cv = cv + mean(max(0, model.xmin - x) + max(0, x - model.xmax) + ...
                   max(0, model.ymin - y) + max(0, y - model.ymax));

    seg = diff(path, 1, 1);
    segLen = sqrt(sum(seg.^2, 2));
    if any(segLen <= 0)
        cv = cv + 1;
    end

    Rmin = 0;
    if isfield(model, 'rmin') && ~isempty(model.rmin)
        Rmin = double(model.rmin);
    elseif isfield(model, 'n') && ~isempty(model.n) && double(model.n) > 0
        pathDiag = norm(double(path(end, :)) - double(path(1, :)));
        Rmin = pathDiag / (3 * double(model.n));
    end
    if Rmin > 0
        cv = cv + mean(max(0, Rmin - segLen));
    end

    stepSize = 1;
    if isfield(model, 'collisionStep') && isnumeric(model.collisionStep) && isfinite(model.collisionStep)
        stepSize = max(1e-3, double(model.collisionStep));
    end
    if ~isfield(model, 'H') || isempty(model.H)
        if any(~isfinite(objs))
            cv = options.nonfiniteViolationBoost;
        end
        return;
    end

    dense = interpolate_path_local(path, stepSize);
    xd = dense(:, 1);
    yd = dense(:, 2);
    zd = dense(:, 3);

    xi = max(1, min(size(model.H, 2), round(xd)));
    yi = max(1, min(size(model.H, 1), round(yd)));
    ground = double(model.H(sub2ind(size(model.H), yi, xi)));
    zRel = zd - ground;

    hmin = get_model_scalar(model, 'zmin', 0);
    hmax = get_model_scalar(model, 'zmax', 300);
    cv = cv + mean(max(0, hmin - zRel) + max(0, zRel - hmax));

    droneSize = get_model_scalar(model, 'droneSize', 1);
    if ~isfield(model, 'droneSize') && isfield(model, 'drone_size')
        droneSize = get_model_scalar(model, 'drone_size', droneSize);
    end
    D = droneSize;
    [centers, radii] = extract_obstacles(model, inf);
    if numel(zRel) >= 2
        collisionDeficit = 0;
        for j = 1:numel(zRel)-1
            minClear = min(zRel(j), zRel(j+1));
            if ~isempty(centers)
                p1 = [xd(j), yd(j)];
                p2 = [xd(j+1), yd(j+1)];
                for k = 1:size(centers, 1)
                    dk = dist_point_to_segment_2d_local(centers(k, :), p1, p2) - radii(k);
                    if dk < minClear
                        minClear = dk;
                    end
                end
            end
            if minClear < D
                collisionDeficit = collisionDeficit + (D - minClear);
            end
        end
        cv = cv + collisionDeficit / max(1, numel(zRel)-1);
    end

    if size(path, 1) >= 3
        turnDeficit = 0;
        for j = 3:size(path, 1)
            v1 = double(path(j-1, 1:2) - path(j-2, 1:2));
            v2 = double(path(j, 1:2) - path(j-1, 1:2));
            l1 = norm(v1);
            l2 = norm(v2);
            if l1 > 1e-12 && l2 > 1e-12
                cosAlpha = dot(v1, v2) / max(1e-12, l1 * l2);
                cosAlpha = max(-1, min(1, cosAlpha));
                alpha = acosd(cosAlpha);
                if alpha < 75
                    turnDeficit = turnDeficit + (75 - alpha) / 75;
                end
            end

            lxy = norm(double(path(j, 1:2) - path(j-1, 1:2)));
            if lxy > 1e-12
                beta = atand(abs(double(path(j, 3) - path(j-1, 3))) / lxy);
                if beta > 60
                    turnDeficit = turnDeficit + (beta - 60) / 60;
                end
            end
        end
        cv = cv + turnDeficit / max(1, size(path, 1)-2);
    end

    if any(~isfinite(objs))
        cv = cv + options.nonfiniteViolationBoost;
    end

    if ~isfinite(cv)
        cv = 1e9;
    elseif cv < 1e-6
        cv = 0;
    end
end

function population = reevaluate_population(population, model, options, aggressive)
    if nargin < 3
        options = [];
    end
    if nargin < 4
        aggressive = false;
    end
    for i = 1:numel(population)
        population(i) = repair_chromosome(population(i), model, options, aggressive);
    end
end

function popObj = population_to_obj(population, M)
    obj = [population.objs];
    popObj = reshape(obj, M, numel(population))';
end

function d = population_diversity(popObj)
    if isempty(popObj)
        d = 0;
        return;
    end

    popObj = popObj(all(isfinite(popObj), 2), :);
    if isempty(popObj)
        d = 0;
        return;
    end

    colMin = min(popObj, [], 1);
    colMax = max(popObj, [], 1);
    denom = colMax - colMin;
    denom(denom == 0) = 1;
    z = (popObj - colMin) ./ denom;
    d = mean(std(z, 0, 1));
end

function hv = safe_hv(popObj, problemIndex, M, hvSamples)
    popObj = popObj(all(isfinite(popObj), 2), :);
    if isempty(popObj)
        hv = 0;
        return;
    end
    try
        hv = calMetric(1, popObj, problemIndex, M, hvSamples, []);
    catch
        hv = 0;
    end
end

function x = clip01(x)
    x = max(0, min(1, x));
end

function ang = wrap_to_pi(ang)
    ang = mod(ang + pi, 2 * pi) - pi;
end

function save_data(filename, data)
    if isstruct(data)
        dt_sv = data;
        save(filename, 'dt_sv');
    else
        gen_hv = data;
        save(filename, 'gen_hv');
    end
end

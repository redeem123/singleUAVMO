function runScore = mopso_single_run(model, params, run, computeMetrics, ...
    problemIndex, resultsPath, M)
    Generations = params.Generations;
    pop = params.pop;

    nVar = model.n; VarSize = [1 nVar];
    VarMin.x = model.xmin; VarMax.x = model.xmax;
    VarMin.y = model.ymin; VarMax.y = model.ymax;
    VarMin.z = model.zmin; VarMax.z = model.zmax;
    VarMin.z = mopso_adjust_min_z(model, VarMin.z, VarMax.z);

    div = 10;
    if isfield(params, 'div')
        div = params.div;
    end
    w = 0.4;
    if isfield(params, 'inertia')
        w = params.inertia;
    end
    if isfield(params, 'w')
        w = params.w;
    end
    metricInterval = 100;

    fprintf('  - Starting MOPSO Run %d/%d\n', run, params.Runs);
    runStart = tic;
    empty_particle = struct('Position',[], 'Velocity',[], 'Cost',[], ...
                            'Best',struct('Position',[], 'Cost',[]), ...
                            'IsDominated',[], 'GridIndex',[], 'GridSubIndex',[]);
    particle = repmat(empty_particle, pop, 1);

    for i = 1:pop
        particle(i).Position.x = unifrnd(VarMin.x, VarMax.x, VarSize);
        particle(i).Position.y = unifrnd(VarMin.y, VarMax.y, VarSize);
        particle(i).Position.z = unifrnd(VarMin.z, VarMax.z, VarSize);

        particle(i).Velocity.x = zeros(VarSize);
        particle(i).Velocity.y = zeros(VarSize);
        particle(i).Velocity.z = zeros(VarSize);

        particle(i).Cost = MOPSO_CostFunction(particle(i).Position, model, []);

        particle(i).Best.Position = particle(i).Position;
        particle(i).Best.Cost = particle(i).Cost;
    end

    archive = mopso_update_archive(particle, pop, div);
    pbest = particle;

    local_gen_hv = [];
    if computeMetrics
        local_gen_hv = zeros(Generations, 2);
    end

    for it = 1:Generations
        if mod(it, 100) == 0 || it == 1
            fprintf('    - Run %d: Iteration %d/%d\n', run, it, Generations);
        end

        if isempty(archive)
            gbest = pbest;
        else
            rep = mopso_rep_selection(archive, pop, div);
            gbest = archive(rep);
        end

        particle = mopso_operator(particle, pbest, gbest, w, VarMin, VarMax);
        for i = 1:pop
            particle(i).Cost = MOPSO_CostFunction(particle(i).Position, model, []);
        end

        if ~isstruct(archive) || isempty(archive)
            archive = particle([]);
        end
        archive = archive(:)'; % ensure row vector for concatenation
        particle = particle(:)'; % keep row shape
        archive = mopso_update_archive([archive, particle], pop, div);
        pbest = mopso_update_pbest(pbest, particle);

        PopObj = mopso_cost_matrix(archive);
        if computeMetrics && ~isempty(PopObj)
            if mod(it, metricInterval) == 0 || it == 1 || it == Generations
                local_gen_hv(it, :) = [calMetric(1, PopObj, problemIndex, M), ...
                    calMetric(2, PopObj, problemIndex, M)];
            elseif it > 1
                local_gen_hv(it, :) = local_gen_hv(it-1, :);
            end
        end
    end

    run_dir = fullfile(resultsPath, sprintf('Run_%d', run));
    if ~isfolder(run_dir)
        mkdir(run_dir);
    end

    if computeMetrics
        mopso_save_data(fullfile(run_dir, 'gen_hv.mat'), local_gen_hv);
    end

    PopObj = mopso_cost_matrix(archive);
    if isempty(PopObj)
        PopObj = mopso_cost_matrix(particle);
    end

    if size(PopObj, 2) ~= M
        if size(PopObj, 1) == M
            PopObj = PopObj';
        end
    end

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

    if ~isempty(archive)
        mopso_save_rep_paths(archive, model, run_dir);
    else
        mopso_save_rep_paths(particle, model, run_dir);
    end
end

function archive = mopso_update_archive(archive, maxSize, div)
    if isempty(archive)
        return;
    end
    archive = archive(:)'; % normalize to row vector
    PopObj = mopso_cost_matrix(archive);
    if isempty(PopObj)
        archive = archive([]);
        return;
    end
    nd = NDSort(PopObj, 1);
    archive = archive(nd == 1);
    archive = archive(:)'; % normalize to row vector
    if numel(archive) > maxSize
        PopObj = mopso_cost_matrix(archive);
        del = mopso_delete(PopObj, numel(archive) - maxSize, div);
        archive(del) = [];
    end
end

function minZ = mopso_adjust_min_z(model, minZ, maxZ)
    if isfield(model, 'safeH') && ~isempty(model.safeH) && isfinite(model.safeH)
        minZ = max(minZ, double(model.safeH));
    else
        D = 1;
        if isfield(model, 'droneSize') && ~isempty(model.droneSize)
            D = double(model.droneSize);
        elseif isfield(model, 'drone_size') && ~isempty(model.drone_size)
            D = double(model.drone_size);
        end
        minZ = max(minZ, D + 1e-3);
    end
    if minZ > maxZ
        minZ = maxZ;
    end
end

function del = mopso_delete(PopObj, K, div)
    N = size(PopObj, 1);
    fmax = max(PopObj, [], 1);
    fmin = min(PopObj, [], 1);
    d = (fmax - fmin) / div;
    GLoc = floor((PopObj - repmat(fmin, N, 1)) ./ repmat(d, N, 1));
    GLoc(GLoc >= div) = div - 1;
    GLoc(isnan(GLoc)) = 0;
    [~, ~, Site] = unique(GLoc, 'rows');
    CrowdG = hist(Site, 1:max(Site));

    del = false(1, N);
    while sum(del) < K
        maxGrid = find(CrowdG == max(CrowdG));
        Grid = maxGrid(randi(length(maxGrid)));
        inGrid = find(Site == Grid);
        p = inGrid(randi(length(inGrid)));
        del(p) = true;
        Site(p) = NaN;
        CrowdG(Grid) = CrowdG(Grid) - 1;
    end
end

function rep = mopso_rep_selection(archive, popSize, div)
    PopObj = mopso_cost_matrix(archive);
    NoP = size(PopObj, 1);
    fmax = max(PopObj, [], 1);
    fmin = min(PopObj, [], 1);
    d = (fmax - fmin) / div;
    fmin = repmat(fmin, NoP, 1);
    d = repmat(d, NoP, 1);
    GLoc = floor((PopObj - fmin) ./ d);
    GLoc(GLoc >= div) = div - 1;
    GLoc(isnan(GLoc)) = 0;
    [~, ~, Site] = unique(GLoc, 'rows');
    CrowdG = hist(Site, 1:max(Site));
    TheGrid = mopso_roulette(popSize, CrowdG);
    rep = zeros(1, popSize);
    for i = 1:popSize
        inGrid = find(Site == TheGrid(i));
        rep(i) = inGrid(randi(length(inGrid)));
    end
end

function idx = mopso_roulette(N, CrowdG)
    Fitness = reshape(CrowdG, 1, []);
    Fitness = Fitness - min(min(Fitness), 0) + 1e-6;
    Fitness = cumsum(1 ./ Fitness);
    Fitness = Fitness ./ max(Fitness);
    idx = arrayfun(@(S)find(rand <= Fitness, 1, 'first'), 1:N);
end

function pbest = mopso_update_pbest(pbest, population)
    pbestObj = mopso_cost_matrix(pbest);
    popObj = mopso_cost_matrix(population);
    if isempty(pbestObj) || isempty(popObj)
        return;
    end
    temp = pbestObj - popObj;
    dominate = any(temp < 0, 2) - any(temp > 0, 2);
    replace = dominate == -1;
    pbest(replace) = population(replace);
    tie = dominate == 0;
    if any(tie)
        pick = rand(sum(tie), 1) < 0.5;
        tieIdx = find(tie);
        pbest(tieIdx(pick)) = population(tieIdx(pick));
    end
end

function population = mopso_operator(population, pbest, gbest, w, VarMin, VarMax)
    if isempty(population)
        return;
    end
    [popDec, popVel, nVar] = mopso_pack(population);
    [pbestDec, ~] = mopso_pack(pbest);
    [gbestDec, ~] = mopso_pack(gbest);

    [N, D] = size(popDec);
    r1 = rand(N, D);
    r2 = rand(N, D);
    offVel = w .* popVel + r1 .* (pbestDec - popDec) + r2 .* (gbestDec - popDec);
    offDec = popDec + offVel;

    [x, y, z] = mopso_unpack(offDec, nVar);
    x = max(VarMin.x, min(VarMax.x, x));
    y = max(VarMin.y, min(VarMax.y, y));
    z = max(VarMin.z, min(VarMax.z, z));

    for i = 1:N
        population(i).Position.x = x(i, :);
        population(i).Position.y = y(i, :);
        population(i).Position.z = z(i, :);
        population(i).Velocity.x = offVel(i, 1:nVar);
        population(i).Velocity.y = offVel(i, nVar+1:2*nVar);
        population(i).Velocity.z = offVel(i, 2*nVar+1:3*nVar);
    end
end

function [dec, vel, nVar] = mopso_pack(population)
    nVar = numel(population(1).Position.x);
    N = numel(population);
    dec = zeros(N, 3*nVar);
    vel = zeros(N, 3*nVar);
    for i = 1:N
        dec(i, 1:nVar) = population(i).Position.x;
        dec(i, nVar+1:2*nVar) = population(i).Position.y;
        dec(i, 2*nVar+1:3*nVar) = population(i).Position.z;
        vel(i, 1:nVar) = population(i).Velocity.x;
        vel(i, nVar+1:2*nVar) = population(i).Velocity.y;
        vel(i, 2*nVar+1:3*nVar) = population(i).Velocity.z;
    end
end

function [x, y, z] = mopso_unpack(dec, nVar)
    x = dec(:, 1:nVar);
    y = dec(:, nVar+1:2*nVar);
    z = dec(:, 2*nVar+1:3*nVar);
end

function PopObj = mopso_cost_matrix(population)
    if isempty(population)
        PopObj = [];
        return;
    end
    PopObj = horzcat(population.Cost)';
    if size(PopObj, 2) ~= numel(population(1).Cost)
        PopObj = PopObj';
    end
end

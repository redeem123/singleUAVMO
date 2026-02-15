function runScore = nmopso_single_run(model, params, run, computeMetrics, ...
    problemIndex, resultsPath, M)
    Generations = params.Generations;
    pop = params.pop;
    ablation = parse_ablation(params);
    representation = ablation.representation;
    if isempty(representation)
        representation = 'SC';
    end
    useSpherical = strcmpi(representation, 'SC');
    useRefLeader = ablation.useReferenceLeader;
    useTwoLayerRef = ablation.useTwoLayerRef;
    atlasCfg = NMOPSO_Utils.BuildAtlasConfig(ablation);
    useAtlasArchive = atlasCfg.enabled && ablation.useRepository;

    nVar = model.n; VarSize = [1 nVar];
    alpha_vel = 0.5;
    if useSpherical
        path_diag = norm(model.start - model.end);
        VarMax.r = 3 * path_diag / nVar; VarMin.r = VarMax.r / 9;
        AngleRange = pi/4;
        VarMin.psi = -AngleRange; VarMax.psi = AngleRange;
        VarMin.phi = -AngleRange; VarMax.phi = AngleRange;
        VelMax.r = alpha_vel*(VarMax.r-VarMin.r); VelMin.r = -VelMax.r;
        VelMax.psi = alpha_vel*(VarMax.psi-VarMin.psi); VelMin.psi = -VelMax.psi;
        VelMax.phi = alpha_vel*(VarMax.phi-VarMin.phi); VelMin.phi = -VelMax.phi;
    else
        VarMin.x = model.xmin; VarMax.x = model.xmax;
        VarMin.y = model.ymin; VarMax.y = model.ymax;
        VarMin.z = model.zmin; VarMax.z = model.zmax;
        VelMax.x = alpha_vel*(VarMax.x-VarMin.x); VelMin.x = -VelMax.x;
        VelMax.y = alpha_vel*(VarMax.y-VarMin.y); VelMin.y = -VelMax.y;
        VelMax.z = alpha_vel*(VarMax.z-VarMin.z); VelMin.z = -VelMax.z;
    end

    nRep = ablation.nRep; wdamp = ablation.wdamp; c1 = ablation.c1; c2 = ablation.c2;
    nGrid = ablation.nGrid; alpha_grid = ablation.alpha_grid; beta = ablation.beta;
    gamma = ablation.gamma; mu = ablation.mu; delta = ablation.delta;
    metricInterval = ablation.metricInterval;
    initMaxTries = 10;
    Z = [];
    if useRefLeader
        try
            Z = build_reference_points(pop, M, useTwoLayerRef);
        catch
            Z = rand(pop, M);
        end
    end

    if isempty(ablation.name)
        fprintf('  - Starting NMOPSO Run %d/%d\n', run, params.Runs);
    else
        fprintf('  - Starting NMOPSO Run %d/%d (%s)\n', run, params.Runs, ablation.name);
    end
    runStart = tic;
    w = ablation.w;
    empty_particle = struct('Position',[], 'Velocity',[], 'Cost',[], ...
                            'Best',struct('Position',[], 'Cost',[]), ...
                            'IsDominated',[], 'GridIndex',[], 'GridSubIndex',[], ...
                            'TopologySignature', [], 'TopologyBin', [], ...
                            'RobustnessScore', [], 'RobustnessBin', [], ...
                            'AtlasCellIndex', []);
    particle = repmat(empty_particle, pop, 1);

    GlobalBest = struct('Position', [], 'Cost', Inf(M, 1));

    isInit = false;
    initTries = 0;
    while ~isInit && initTries < initMaxTries
        initTries = initTries + 1;
        for i = 1:pop
            if useSpherical
                particle(i).Position.r = unifrnd(VarMin.r, VarMax.r, VarSize);
                particle(i).Position.psi = unifrnd(VarMin.psi, VarMax.psi, VarSize);
                particle(i).Position.phi = unifrnd(VarMin.phi, VarMax.phi, VarSize);

                particle(i).Velocity.r = zeros(VarSize);
                particle(i).Velocity.psi = zeros(VarSize);
                particle(i).Velocity.phi = zeros(VarSize);
            else
                particle(i).Position.x = unifrnd(VarMin.x, VarMax.x, VarSize);
                particle(i).Position.y = unifrnd(VarMin.y, VarMax.y, VarSize);
                particle(i).Position.z = unifrnd(VarMin.z, VarMax.z, VarSize);

                particle(i).Velocity.x = zeros(VarSize);
                particle(i).Velocity.y = zeros(VarSize);
                particle(i).Velocity.z = zeros(VarSize);
            end

            cart_sol = NMOPSO_Utils.PositionToCart(particle(i).Position, model, representation);
            particle(i).Cost = NMOPSO_CostFunction(cart_sol, model, []);
            if useAtlasArchive
                particle(i) = NMOPSO_Utils.UpdateAtlasMetadata( ...
                    particle(i), model, representation, atlasCfg, cart_sol);
            end

            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.Cost = particle(i).Cost;

            if NMOPSO_Utils.Dominates(particle(i).Best, GlobalBest)
                GlobalBest = particle(i).Best;
                isInit = true;
            end
        end
    end
    if ~isInit
        allCosts = horzcat(particle.Cost)';
        sumCosts = sum(allCosts, 2);
        sumCosts(~isfinite(sumCosts)) = inf;
        [~, bestIdx] = min(sumCosts);
        GlobalBest = particle(bestIdx).Best;
    end

    particle = NMOPSO_Utils.DetermineDomination(particle);
    rep = [];
    Grid = [];
    if ablation.useRepository
        rep = particle(~[particle.IsDominated]);
        if isempty(rep)
            rep = particle;
        end
        if ablation.useGrid
            Grid = NMOPSO_Utils.CreateGrid(rep, nGrid, alpha_grid);
            for i_grid = 1:numel(rep)
                rep(i_grid) = NMOPSO_Utils.FindGridIndex(rep(i_grid), Grid);
            end
        end
        if useAtlasArchive
            rep = NMOPSO_Utils.RefreshAtlasCellIndex(rep, atlasCfg);
        end
    end

    local_gen_hv = [];
    if computeMetrics
        local_gen_hv = zeros(Generations, 2);
    end

    for it = 1:Generations
        if mod(it, 50) == 0 || it == 1 || it == Generations
            fprintf('    - Run %d: Iteration %d/%d\n', run, it, Generations);
        end

        leaderIdx = [];
        if useRefLeader && ablation.useRepository && ~isempty(rep)
            leaderIdx = NMOPSO_Utils.SelectLeaderRef(rep, Z, pop);
        end

        for i = 1:pop
            leader = GlobalBest;
            if ablation.useRepository && ~isempty(rep)
                if useRefLeader && ~isempty(leaderIdx)
                    leader = rep(leaderIdx(i));
                elseif useAtlasArchive
                    leader = NMOPSO_Utils.SelectLeaderAtlas( ...
                        rep, beta, atlasCfg.objectiveWeight, atlasCfg.atlasWeight);
                elseif ablation.useGrid
                    leader = NMOPSO_Utils.SelectLeader(rep, beta);
                else
                    leader = rep(randi(numel(rep)));
                end
            end

            if useSpherical
                particle(i).Velocity.r = w*particle(i).Velocity.r ...
                    + c1*rand(VarSize).*(particle(i).Best.Position.r - particle(i).Position.r) ...
                    + c2*rand(VarSize).*(leader.Position.r - particle(i).Position.r);
                particle(i).Velocity.r = max(VelMin.r, min(VelMax.r, particle(i).Velocity.r));
                particle(i).Position.r = particle(i).Position.r + particle(i).Velocity.r;

                out_r = (particle(i).Position.r < VarMin.r | particle(i).Position.r > VarMax.r);
                particle(i).Velocity.r(out_r) = -particle(i).Velocity.r(out_r);
                particle(i).Position.r = max(VarMin.r, min(VarMax.r, particle(i).Position.r));

                particle(i).Velocity.psi = w*particle(i).Velocity.psi ...
                    + c1*rand(VarSize).*(particle(i).Best.Position.psi - particle(i).Position.psi) ...
                    + c2*rand(VarSize).*(leader.Position.psi - particle(i).Position.psi);
                particle(i).Velocity.psi = max(VelMin.psi, min(VelMax.psi, particle(i).Velocity.psi));
                particle(i).Position.psi = particle(i).Position.psi + particle(i).Velocity.psi;

                out_psi = (particle(i).Position.psi < VarMin.psi | particle(i).Position.psi > VarMax.psi);
                particle(i).Velocity.psi(out_psi) = -particle(i).Velocity.psi(out_psi);
                particle(i).Position.psi = max(VarMin.psi, min(VarMax.psi, particle(i).Position.psi));

                particle(i).Velocity.phi = w*particle(i).Velocity.phi ...
                    + c1*rand(VarSize).*(particle(i).Best.Position.phi - particle(i).Position.phi) ...
                    + c2*rand(VarSize).*(leader.Position.phi - particle(i).Position.phi);
                particle(i).Velocity.phi = max(VelMin.phi, min(VelMax.phi, particle(i).Velocity.phi));
                particle(i).Position.phi = particle(i).Position.phi + particle(i).Velocity.phi;

                out_phi = (particle(i).Position.phi < VarMin.phi | particle(i).Position.phi > VarMax.phi);
                particle(i).Velocity.phi(out_phi) = -particle(i).Velocity.phi(out_phi);
                particle(i).Position.phi = max(VarMin.phi, min(VarMax.phi, particle(i).Position.phi));
            else
                particle(i).Velocity.x = w*particle(i).Velocity.x ...
                    + c1*rand(VarSize).*(particle(i).Best.Position.x - particle(i).Position.x) ...
                    + c2*rand(VarSize).*(leader.Position.x - particle(i).Position.x);
                particle(i).Velocity.x = max(VelMin.x, min(VelMax.x, particle(i).Velocity.x));
                particle(i).Position.x = particle(i).Position.x + particle(i).Velocity.x;

                out_x = (particle(i).Position.x < VarMin.x | particle(i).Position.x > VarMax.x);
                particle(i).Velocity.x(out_x) = -particle(i).Velocity.x(out_x);
                particle(i).Position.x = max(VarMin.x, min(VarMax.x, particle(i).Position.x));

                particle(i).Velocity.y = w*particle(i).Velocity.y ...
                    + c1*rand(VarSize).*(particle(i).Best.Position.y - particle(i).Position.y) ...
                    + c2*rand(VarSize).*(leader.Position.y - particle(i).Position.y);
                particle(i).Velocity.y = max(VelMin.y, min(VelMax.y, particle(i).Velocity.y));
                particle(i).Position.y = particle(i).Position.y + particle(i).Velocity.y;

                out_y = (particle(i).Position.y < VarMin.y | particle(i).Position.y > VarMax.y);
                particle(i).Velocity.y(out_y) = -particle(i).Velocity.y(out_y);
                particle(i).Position.y = max(VarMin.y, min(VarMax.y, particle(i).Position.y));

                particle(i).Velocity.z = w*particle(i).Velocity.z ...
                    + c1*rand(VarSize).*(particle(i).Best.Position.z - particle(i).Position.z) ...
                    + c2*rand(VarSize).*(leader.Position.z - particle(i).Position.z);
                particle(i).Velocity.z = max(VelMin.z, min(VelMax.z, particle(i).Velocity.z));
                particle(i).Position.z = particle(i).Position.z + particle(i).Velocity.z;

                out_z = (particle(i).Position.z < VarMin.z | particle(i).Position.z > VarMax.z);
                particle(i).Velocity.z(out_z) = -particle(i).Velocity.z(out_z);
                particle(i).Position.z = max(VarMin.z, min(VarMax.z, particle(i).Position.z));
            end

            cart_sol = NMOPSO_Utils.PositionToCart(particle(i).Position, model, representation);
            particle(i).Cost = NMOPSO_CostFunction(cart_sol, model, []);

            if ablation.useMutation
                if ablation.useAdaptiveMutation
                    mutationProb = (1-(it-1)/(Generations-1))^(1/mu);
                else
                    mutationProb = ablation.mutationProb;
                end
            else
                mutationProb = 0;
            end
            if rand < mutationProb
                NewSol_Local = struct();
                pm = [];
                if ablation.useRegionMutation && ablation.useRepository && ~isempty(rep)
                    pm = rep;
                end
                NewSol_Local.Position = NMOPSO_Utils.Mutate(particle(i), pm, delta, VarMax, VarMin, representation);
                cart_new = NMOPSO_Utils.PositionToCart(NewSol_Local.Position, model, representation);
                NewSol_Local.Cost = NMOPSO_CostFunction(cart_new, model, []);
                if NMOPSO_Utils.Dominates(NewSol_Local, particle(i))
                    particle(i).Position = NewSol_Local.Position;
                    particle(i).Cost = NewSol_Local.Cost;
                elseif ~NMOPSO_Utils.Dominates(particle(i), NewSol_Local)
                    if rand < 0.5
                        particle(i).Position = NewSol_Local.Position;
                        particle(i).Cost = NewSol_Local.Cost;
                    end
                end
            end

            if useAtlasArchive
                particle(i) = NMOPSO_Utils.UpdateAtlasMetadata( ...
                    particle(i), model, representation, atlasCfg, []);
            end

            if NMOPSO_Utils.Dominates(particle(i), particle(i).Best)
                particle(i).Best.Position = particle(i).Position;
                particle(i).Best.Cost = particle(i).Cost;
            elseif ~NMOPSO_Utils.Dominates(particle(i).Best, particle(i))
                if rand < 0.5
                    particle(i).Best.Position = particle(i).Position;
                    particle(i).Best.Cost = particle(i).Cost;
                end
            end

            if ~ablation.useRepository && NMOPSO_Utils.Dominates(particle(i), GlobalBest)
                GlobalBest = particle(i);
            end
        end

        if ablation.useRepository
            particle = NMOPSO_Utils.DetermineDomination(particle);
            rep = [rep; particle(~[particle.IsDominated])];
            rep = NMOPSO_Utils.DetermineDomination(rep);
            rep = rep(~[rep.IsDominated]);
            if useAtlasArchive
                rep = NMOPSO_Utils.RefreshAtlasCellIndex(rep, atlasCfg);
            end

            if ablation.useGrid
                Grid = NMOPSO_Utils.CreateGrid(rep, nGrid, alpha_grid);
                for i_g = 1:numel(rep)
                    rep(i_g) = NMOPSO_Utils.FindGridIndex(rep(i_g), Grid);
                end

                while numel(rep) > nRep
                    if useAtlasArchive
                        rep = NMOPSO_Utils.DeleteOneRepMemberAtlas( ...
                            rep, gamma, atlasCfg.objectiveWeight, atlasCfg.atlasWeight);
                    else
                        rep = NMOPSO_Utils.DeleteOneRepMember(rep, gamma);
                    end
                end
            else
                if numel(rep) > nRep
                    if useAtlasArchive
                        while numel(rep) > nRep
                            rep = NMOPSO_Utils.DeleteOneRepMemberAtlas( ...
                                rep, gamma, atlasCfg.objectiveWeight, atlasCfg.atlasWeight);
                        end
                    else
                        rep = rep(randperm(numel(rep), nRep));
                    end
                end
            end
        end

        PopObj = [];
        if ablation.useRepository && ~isempty(rep)
            PopObj = horzcat(rep.Cost)';
        else
            PopObj = horzcat(particle.Cost)';
        end
        if size(PopObj, 2) ~= M
            if size(PopObj, 1) == M
                PopObj = PopObj';
            end
        end

        if computeMetrics
            if mod(it, metricInterval) == 0 || it == 1 || it == Generations
                local_gen_hv(it, :) = [calMetric(1, PopObj, problemIndex, M), ...
                    calMetric(2, PopObj, problemIndex, M)];
            elseif it > 1
                local_gen_hv(it, :) = local_gen_hv(it-1, :);
            end
        end
        w = w * wdamp;
    end

    run_dir = fullfile(resultsPath, sprintf('Run_%d', run));
    if ~isfolder(run_dir)
        mkdir(run_dir);
    end
    if computeMetrics
        nmopso_save_data(fullfile(run_dir, 'gen_hv.mat'), local_gen_hv);
    end

    if ablation.useRepository && ~isempty(rep)
        PopObj = horzcat(rep.Cost)';
    else
        PopObj = horzcat(particle.Cost)';
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

    if ablation.useRepository && ~isempty(rep)
        nmopso_save_rep_paths(rep, model, run_dir, representation);
    else
        nmopso_save_rep_paths(particle, model, run_dir, representation);
    end
end

function ablation = parse_ablation(params)
    ablation = struct();
    ablation.name = '';
    ablation.useRepository = true;
    ablation.useGrid = true;
    ablation.useMutation = true;
    ablation.useAdaptiveMutation = true;
    ablation.useRegionMutation = true;
    ablation.mutationProb = 0.1;
    ablation.representation = 'SC';
    ablation.useReferenceLeader = false;
    ablation.useTwoLayerRef = false;
    ablation.nRep = 50;
    ablation.nGrid = 5;
    ablation.alpha_grid = 0.1;
    ablation.beta = 2;
    ablation.gamma = 2;
    ablation.w = 1;
    ablation.wdamp = 0.98;
    ablation.c1 = 1.5;
    ablation.c2 = 1.5;
    ablation.mu = 0.5;
    ablation.delta = 20;
    ablation.metricInterval = 100;
    ablation.useTopologyRobustArchive = false;
    ablation.atlasTopologyBins = 24;
    ablation.atlasRobustBins = 4;
    ablation.atlasMaxObstacles = 3;
    ablation.atlasHashLevels = 6;
    ablation.atlasObjectiveWeight = 0.5;
    ablation.atlasTopologyWeight = 0.5;

    if isfield(params, 'ablation') && ~isempty(params.ablation)
        in = params.ablation;
        if isfield(in, 'name'), ablation.name = in.name; end
        if isfield(in, 'useRepository'), ablation.useRepository = logical(in.useRepository); end
        if isfield(in, 'useGrid'), ablation.useGrid = logical(in.useGrid); end
        if isfield(in, 'useMutation'), ablation.useMutation = logical(in.useMutation); end
        if isfield(in, 'useAdaptiveMutation'), ablation.useAdaptiveMutation = logical(in.useAdaptiveMutation); end
        if isfield(in, 'useRegionMutation'), ablation.useRegionMutation = logical(in.useRegionMutation); end
        if isfield(in, 'mutationProb') && isnumeric(in.mutationProb) && isfinite(in.mutationProb)
            ablation.mutationProb = max(0, min(1, in.mutationProb));
        end
        if isfield(in, 'representation') && ~isempty(in.representation)
            ablation.representation = normalize_representation(in.representation);
        end
        if isfield(in, 'useReferenceLeader')
            ablation.useReferenceLeader = logical(in.useReferenceLeader);
        end
        if isfield(in, 'useTwoLayerRef')
            ablation.useTwoLayerRef = logical(in.useTwoLayerRef);
        end
        if isfield(in, 'nRep'), ablation.nRep = in.nRep; end
        if isfield(in, 'nGrid'), ablation.nGrid = in.nGrid; end
        if isfield(in, 'alpha_grid'), ablation.alpha_grid = in.alpha_grid; end
        if isfield(in, 'beta'), ablation.beta = in.beta; end
        if isfield(in, 'gamma'), ablation.gamma = in.gamma; end
        if isfield(in, 'w'), ablation.w = in.w; end
        if isfield(in, 'wdamp'), ablation.wdamp = in.wdamp; end
        if isfield(in, 'c1'), ablation.c1 = in.c1; end
        if isfield(in, 'c2'), ablation.c2 = in.c2; end
        if isfield(in, 'mu'), ablation.mu = in.mu; end
        if isfield(in, 'delta'), ablation.delta = in.delta; end
        if isfield(in, 'metricInterval'), ablation.metricInterval = in.metricInterval; end
        if isfield(in, 'useTopologyRobustArchive')
            ablation.useTopologyRobustArchive = logical(in.useTopologyRobustArchive);
        end
        if isfield(in, 'atlasTopologyBins'), ablation.atlasTopologyBins = in.atlasTopologyBins; end
        if isfield(in, 'atlasRobustBins'), ablation.atlasRobustBins = in.atlasRobustBins; end
        if isfield(in, 'atlasMaxObstacles'), ablation.atlasMaxObstacles = in.atlasMaxObstacles; end
        if isfield(in, 'atlasHashLevels'), ablation.atlasHashLevels = in.atlasHashLevels; end
        if isfield(in, 'atlasObjectiveWeight'), ablation.atlasObjectiveWeight = in.atlasObjectiveWeight; end
        if isfield(in, 'atlasTopologyWeight'), ablation.atlasTopologyWeight = in.atlasTopologyWeight; end
    end
    if isfield(params, 'representation') && ~isempty(params.representation)
        ablation.representation = normalize_representation(params.representation);
    end
    if isfield(params, 'useReferenceLeader')
        ablation.useReferenceLeader = logical(params.useReferenceLeader);
    end
    if isfield(params, 'useTwoLayerRef')
        ablation.useTwoLayerRef = logical(params.useTwoLayerRef);
    end
    if isfield(params, 'nRep'), ablation.nRep = params.nRep; end
    if isfield(params, 'nGrid'), ablation.nGrid = params.nGrid; end
    if isfield(params, 'alpha_grid'), ablation.alpha_grid = params.alpha_grid; end
    if isfield(params, 'beta'), ablation.beta = params.beta; end
    if isfield(params, 'gamma'), ablation.gamma = params.gamma; end
    if isfield(params, 'w'), ablation.w = params.w; end
    if isfield(params, 'wdamp'), ablation.wdamp = params.wdamp; end
    if isfield(params, 'c1'), ablation.c1 = params.c1; end
    if isfield(params, 'c2'), ablation.c2 = params.c2; end
    if isfield(params, 'mu'), ablation.mu = params.mu; end
    if isfield(params, 'delta'), ablation.delta = params.delta; end
    if isfield(params, 'metricInterval'), ablation.metricInterval = params.metricInterval; end
    if isfield(params, 'useTopologyRobustArchive')
        ablation.useTopologyRobustArchive = logical(params.useTopologyRobustArchive);
    end
    if isfield(params, 'atlasTopologyBins'), ablation.atlasTopologyBins = params.atlasTopologyBins; end
    if isfield(params, 'atlasRobustBins'), ablation.atlasRobustBins = params.atlasRobustBins; end
    if isfield(params, 'atlasMaxObstacles'), ablation.atlasMaxObstacles = params.atlasMaxObstacles; end
    if isfield(params, 'atlasHashLevels'), ablation.atlasHashLevels = params.atlasHashLevels; end
    if isfield(params, 'atlasObjectiveWeight'), ablation.atlasObjectiveWeight = params.atlasObjectiveWeight; end
    if isfield(params, 'atlasTopologyWeight'), ablation.atlasTopologyWeight = params.atlasTopologyWeight; end
end

function rep = normalize_representation(value)
    if isnumeric(value)
        if value == 0
            rep = 'CC';
        else
            rep = 'SC';
        end
        return;
    end
    if isstring(value)
        value = char(value);
    end
    value = upper(strtrim(value));
    if any(strcmp(value, {'SC', 'SPHERICAL'}))
        rep = 'SC';
    elseif any(strcmp(value, {'CC', 'CARTESIAN'}))
        rep = 'CC';
    else
        rep = 'SC';
    end
end

function Z = build_reference_points(N, M, useTwoLayer)
    if nargin < 3
        useTwoLayer = false;
    end
    [Z1, ~] = UniformPoint(N, M);
    if ~useTwoLayer
        Z = Z1;
        return;
    end
    n2 = max(1, floor(N / 2));
    [Z2, ~] = UniformPoint(n2, M);
    Z2 = Z2 / 2 + 1 / (2 * M);
    Z = [Z1; Z2];
end

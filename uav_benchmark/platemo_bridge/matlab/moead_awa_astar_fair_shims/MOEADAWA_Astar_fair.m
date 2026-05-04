function [Score,BestPopulation,gen_hv] = MOEADAWA_Astar_fair(M,model,pop,Generations,nVar,problemIndex)
% Fair benchmark shim for the author MOEA/D-AWA + A* implementation.
% The evolutionary flow and A*-guided operators remain in MATLAB; objective
% and constraint evaluation are routed through BridgeEvaluatePathPopulation.

    %#ok<INUSD>
    M = double(M);
    pop = double(pop);
    Generations = double(Generations);
    nVar = double(nVar);
    model.n = nVar;
    MinValue = [model.xmin,model.ymin,model.zmin];
    MaxValue = [model.xmax,model.ymax,model.zmax];
    boundary = [MaxValue;MinValue];
    Score = zeros(1,2);
    gen_hv = zeros(0,2);

    rate_update_weight = 0.05;
    rate_evol = 0.8;
    wag = 100;
    [W,pop] = UniformPoint(pop,M);
    W = 1./W./repmat(sum(1./W,2),1,size(W,2));
    nr = max(1,ceil(pop/100));
    nEP = max(1,ceil(pop*1.5));
    T = min(pop,max(2,ceil(pop/10)));

    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:T);

    for i = 1 : pop
        population(i) = Chromosome(model); %#ok<AGROW>
        population(i) = initialize(population(i),model);
    end
    population = BridgeEvaluatePathPopulation(population);

    objs = [population.objs];
    obj = reshape(objs,M,length(population))';
    Z = min(obj,[],1);
    Pi = ones(pop,1);
    oldObj = max(abs((obj-repmat(Z,pop,1)).*W),[],2);
    oldObj(oldObj == 0) = eps;

    EP = [];
    gen = 0;
    while gen < Generations
        if ~mod(ceil(gen*pop/pop),10)
            objs = [population.objs];
            obj = reshape(objs,M,length(population))';
            newObj = max(abs((obj-repmat(Z,pop,1)).*W),[],2);
            DELTA = (oldObj-newObj)./oldObj;
            DELTA(~isfinite(DELTA)) = 0;
            Temp = DELTA <= 0.001;
            Pi(~Temp) = 1;
            Pi(Temp) = (0.95+0.05*DELTA(Temp)/0.001).*Pi(Temp);
            oldObj = newObj;
            oldObj(oldObj == 0) = eps;
        end
        offspringCount = 0;
        clear offspring;
        for subgeneration = 1 : 5
            %#ok<NASGU>
            Boundary = find(sum(W<1e-3,2)==1)';
            quota = max(0,floor(pop/5)-length(Boundary));
            if quota > 0
                I = [Boundary,TournamentSelection(10,quota,-Pi)];
            else
                I = Boundary;
            end
            if isempty(I)
                I = randperm(pop,min(pop,max(1,floor(pop/5))));
            end
            for i = 1 : length(I)
                if rand < 1
                    P = B(I(i),randperm(size(B,2)));
                else
                    P = randperm(pop);
                end
                if length(P) < 2
                    P = [P, P(1)];
                end
                child = F_operator(population(I(i)),population(P(1:2)),boundary,model,gen/Generations);
                offspringCount = offspringCount + 1;
                if offspringCount == 1
                    offspring = child; %#ok<NASGU>
                else
                    offspring(offspringCount) = child; %#ok<AGROW>
                end
                Z = min(Z,child.objs);

                pop_obj = [population(P).objs];
                pop_obj = reshape(pop_obj,M,length(P))';
                g_old = max(abs(pop_obj-repmat(Z,length(P),1)).*W(P,:),[],2);
                g_new = max(repmat(abs(child.objs-Z),length(P),1).*W(P,:),[],2);
                replaceable = find(g_old>=g_new,nr);
                if ~isempty(replaceable)
                    population(P(replaceable)) = child;
                end
            end
        end
        if gen*pop >= rate_evol*Generations*pop && exist('offspring','var') && ~isempty(offspring)
            if isempty(EP)
                EP = updateEP(population,offspring,nEP);
            else
                EP = updateEP(EP,offspring,nEP);
            end
            if ~isempty(EP) && ~mod(ceil(gen*pop/pop),wag/5)
                [population,W] = updateWeight(population,W,Z,EP,rate_update_weight*pop,M);
            end
        end
        gen = gen + 1;
    end

    BestPopulation = population;
end

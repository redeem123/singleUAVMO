function [Score,BestPopulation] = MOEA_2DE(M,model,pop,Gen,F,pc,pm,nVar,problemIndex)
% Fair bridge version of the official MOEA-2DE loop.
% The algorithm mechanics are kept in MATLAB; path objective evaluation is
% batched through the Python UAV benchmark evaluator.

eval = Gen * pop;
pro = 0.3;
adjustGeneration = 1;

for i = 1 : pop
    population(i) = Chromosome(model); %#ok<AGROW>
    child(i) = Chromosome(model); %#ok<AGROW>
    population(i) = initialize(population(i),model);
end
population = BridgeEvaluatePathPopulation(population);
[~,FrontNo,CrowdDis] = EnvironmentalSelection(population,pop,pop,M);
dim = size(population(1).rnvec,1);
Score(1,1) = 0;
Score(2,1) = 0;
BestPopulation = population;

strongDim = zeros(1,dim);
b = strongDim;
parent = Chromosome(model);
parent.rnvec(1,:) = model.start;
parent.rnvec(model.n,:) = model.end;
parent.rnvec(2:dim-1,1) = [2:dim-1]*model.xmax/model.n;
parent.rnvec(2:dim-1,2) = [2:dim-1]*model.xmax/model.n;
parent.path = testBspline([parent.rnvec(:,1)';parent.rnvec(:,2)'],model.xmax)';
parent = adjust_constraint_turning_angle(parent,model);
parent = BridgeEvaluatePathPopulation(parent);

times = 3;
varRange = times*model.xmax/model.n;
for j = 1 : dim - 2
    testPopulation(j) = Chromosome(model); %#ok<AGROW>
    testPopulation(j).rnvec = parent.rnvec;
    rnd = randi(2);
    testPopulation(j).rnvec(j+1,rnd) = parent.rnvec(j+1,rnd) + varRange;
    testPopulation(j).rnvec = Evolve.check_boundary(testPopulation(j).rnvec,j+1,model);
    testPopulation(j).path = testBspline([testPopulation(j).rnvec(:,1)';testPopulation(j).rnvec(:,2)'],model.xmax)';
    testPopulation(j) = adjust_constraint_turning_angle(testPopulation(j),model);
end
testPopulation = BridgeEvaluatePathPopulation(testPopulation);
for j = 1 : dim - 2
    dimDiff(j,1) = abs(parent.objs(1) - testPopulation(j).objs(1)); %#ok<AGROW>
    dimDiff(j,2) = abs(parent.objs(2) - testPopulation(j).objs(2)); %#ok<AGROW>
end
v = mean(dimDiff);
a = find(dimDiff(:,1)>v(1));
b(a)=1;
a = find(dimDiff(:,2)>v(2));
b(a)=1;
strongDim = b;
strongDim(1)=0;
strongDim(end)=0;
eval = eval - (dim-1);

i = 0;
while 1
    lastpopulation = population;
    eliteCluster = population(FrontNo == 1);
    MatingPool = TournamentSelection(2,pop,FrontNo,-CrowdDis);
    childEvalCount = 0;
    for j = 1 : 2 : pop-1
        if rand(1)<pro
            parent1 = eliteCluster(randi(length(eliteCluster)));
            [child(j)] = Evolve.dimExplore(parent1,dim,model,strongDim,F);
            parent2 = eliteCluster(randi(length(eliteCluster)));
            [child(j+1)] = Evolve.dimExplore(parent2,dim,model,strongDim,F);
        else
            [child(j),child(j+1)] = Evolve.binary_crossover(population(MatingPool(j)),population(MatingPool(j+1)),nVar-1,pc,randi(2));
            if rand < pm
                child(j) = Evolve.mutation(child(j),nVar,model);
            end
            if rand < pm
                child(j+1) = Evolve.mutation(child(j+1),nVar,model);
            end
        end
        child(j).rnvec = sortrows(child(j).rnvec,randi(2));
        child(j+1).rnvec = sortrows(child(j+1).rnvec,randi(2));
        child(j).path = testBspline([child(j).rnvec(:,1)';child(j).rnvec(:,2)'],model.xmax)';
        child(j) = adjust_constraint_turning_angle(child(j),model);
        child(j+1).path = testBspline([child(j+1).rnvec(:,1)';child(j+1).rnvec(:,2)'],model.xmax)';
        child(j+1) = adjust_constraint_turning_angle(child(j+1),model);
        childEvalCount = j + 1;

        i = i + 2;
        if i >= eval
            break;
        end
    end
    evaluatedChild = BridgeEvaluatePathPopulation(child(1:childEvalCount));
    [population,FrontNo,CrowdDis] = EnvironmentalSelection([population,evaluatedChild],pop,pop+childEvalCount,M);

    obj = [population.objs];
    PopObj = reshape(obj,M,length(population))';
    g = ceil(i/pop + 1);
    difference(g) = KLD(dim,lastpopulation,population); %#ok<AGROW>
    if g > 1
        if mod(g-1,adjustGeneration) == 0
            j = g-adjustGeneration;
            upGap = 0;
            downGap = 0;
            while j < g
                if difference(j+1) - difference(j) > 0
                    upGap = upGap + difference(j+1) - difference(j);
                else
                    downGap = downGap + difference(j) - difference(j+1);
                end
                j = j + 1;
            end
            if upGap > downGap
                pro = 1;
            else
                pro = 0;
            end
        end
    end

    for j = 1 : 2
        [Score(j,g)] = calMetirc(j,PopObj,problemIndex); %#ok<AGROW>
        if Score(j,g) < Score(j,g-1)
            Score(j,g) = Score(j,g-1);
        else
            BestPopulation = population;
        end
    end

    X = sprintf('generation:%d',g);
    disp(X);
    if i >= eval
        break;
    end
end
end

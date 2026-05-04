function state = stateCal(Population, Fitness)
% Four-objective-compatible EMMOP state summary for fair benchmark wiring.
% Keeps the official DQN input length at 10 while summarizing benchmark
% objective vectors instead of assuming exactly two paper objectives.

    state = zeros(10, 1);
    objVal = double(Population.objs);
    consVal = double(Population.cons);
    decsVal = double(Population.decs);
    consVal(consVal < 0) = 0;
    N = length(Population);

    if ~isequal(Fitness,0)
        Off_Fitness = CalFitness(Population.objs, Population.cons);
        fitDen = max(Fitness) - min(Fitness);
        offDen = max(Off_Fitness) - min(Off_Fitness);
        if fitDen <= eps
            Fitness = zeros(size(Fitness));
        else
            Fitness = (Fitness - min(Fitness)) ./ fitDen;
        end
        if offDen <= eps
            Off_Fitness = zeros(size(Off_Fitness));
        else
            Off_Fitness = (Off_Fitness - min(Off_Fitness)) ./ offDen;
        end
        delta_fitness = min(Fitness) - min(Off_Fitness);
        stag = double(delta_fitness <= 0);
    else
        delta_fitness = 1e-3;
        stag = 0;
    end

    objStd = std(objVal,0,1);
    objStd(objStd <= eps) = 1;
    objVal = (objVal - mean(objVal,1)) ./ objStd;
    objVal(~isfinite(objVal)) = 0;

    if all(consVal == 0,'all')
        consNorm = zeros(size(consVal));
    else
        consStd = std(consVal,0,1);
        consStd(consStd <= eps) = 1;
        consNorm = (consVal - mean(consVal,1)) ./ consStd;
        consNorm(~isfinite(consNorm)) = 0;
    end

    state(1) = max(objVal,[],'all');
    state(2) = min(objVal,[],'all');
    state(3) = mean(objVal,'all');
    state(4) = median(objVal,'all');
    state(5) = mean(std(objVal,0,1));
    state(6) = mean(max(objVal,[],1) - min(objVal,[],1));
    state(7) = mean(consNorm,'all');
    state(8) = sum(all(consVal <= 0,2)) / max(1,N);

    DistMat = pdist2(objVal, objVal);
    DistMat(logical(eye(length(DistMat)))) = inf;
    DistMat = sort(DistMat, 2);
    MinDist = DistMat(:, 1);
    AvgDist = mean(MinDist);
    if length(MinDist) > 1
        state(9) = sqrt(sum((MinDist - AvgDist).^2) / (length(MinDist) - 1));
    else
        state(9) = 0;
    end

    DistDec = 0;
    for i = 1:N-1
        for j = i+1:N
            DistDec = DistDec + norm(decsVal(i, :) - decsVal(j, :));
        end
    end
    state(10) = DistDec / max(1,(N * (N-1) / 2));
    state(~isfinite(state)) = 0;
end

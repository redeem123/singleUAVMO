function score = PD(Population)
% Numeric-compatible population diversity shim for old reference code.

    if isa(Population,'INDIVIDUAL')
        PopObj = Population.objs;
    else
        PopObj = double(Population);
    end
    if size(PopObj,1) < 2
        score = 0;
        return;
    end
    DistMat = pdist2(PopObj,PopObj);
    DistMat(DistMat == 0) = inf;
    nearest = min(DistMat,[],2);
    nearest(~isfinite(nearest)) = 0;
    score = mean(nearest);
end

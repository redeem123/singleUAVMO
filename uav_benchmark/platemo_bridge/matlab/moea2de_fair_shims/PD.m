function Score = PD(PopObj)
% Simple diversity metric used only for MOEA-2DE's progress bookkeeping.

    PopObj = double(PopObj);
    if size(PopObj,1) < 2
        Score = 0;
        return;
    end
    D = pdist2(PopObj,PopObj,'euclidean');
    D(logical(eye(size(D)))) = inf;
    nearest = min(D,[],2);
    nearest(~isfinite(nearest)) = 0;
    Score = mean(nearest);
end

function Population = EnvironmentalSelection_NSGAIII(Population, N, Z, Zmin)
% The environmental selection of NSGA-III (adapted for UAV path planning)

    if isempty(Population)
        return;
    end

    if isempty(Zmin)
        Zmin = ones(1, size(Z, 2));
    end

    PopSize = numel(Population);
    M = numel(Population(1).objs);
    obj = [Population.objs];
    PopObj = reshape(obj, M, PopSize)';

    PopCon = extract_constraints(Population);

    %% Non-dominated sorting
    if isempty(PopCon)
        [FrontNo, MaxFNo] = NDSort(PopObj, N);
    else
        [FrontNo, MaxFNo] = NDSort(PopObj, PopCon, N);
    end
    Next = FrontNo < MaxFNo;

    %% Select the solutions in the last front
    Last = find(FrontNo == MaxFNo);
    if sum(Next) < N && ~isempty(Last)
        Choose = LastSelection(PopObj(Next, :), PopObj(Last, :), N - sum(Next), Z, Zmin);
        Next(Last(Choose)) = true;
    end

    Population = Population(Next);
end

function Choose = LastSelection(PopObj1, PopObj2, K, Z, Zmin)
% Select part of the solutions in the last front

    if K <= 0
        Choose = false(1, size(PopObj2, 1));
        return;
    end

    PopObj = [PopObj1; PopObj2] - repmat(Zmin, size(PopObj1, 1) + size(PopObj2, 1), 1);
    [N, M]  = size(PopObj);
    N1      = size(PopObj1, 1);
    N2      = size(PopObj2, 1);
    NZ      = size(Z, 1);

    %% Normalization
    Extreme = zeros(1, M);
    w = zeros(M) + 1e-6 + eye(M);
    for i = 1 : M
        [~, Extreme(i)] = min(max(PopObj ./ repmat(w(i, :), N, 1), [], 2));
    end
    Hyperplane = PopObj(Extreme, :) \ ones(M, 1);
    a = 1 ./ Hyperplane;
    if any(isnan(a)) || any(isinf(a))
        a = max(PopObj, [], 1)';
    end
    PopObj = PopObj ./ repmat(a', N, 1);

    %% Associate each solution with one reference point
    Cosine   = 1 - pdist2(PopObj, Z, 'cosine');
    Distance = repmat(sqrt(sum(PopObj.^2, 2)), 1, NZ) .* sqrt(1 - Cosine.^2);
    [d, pi] = min(Distance', [], 1);

    %% Calculate the number of associated solutions except for the last front
    rho = hist(pi(1:N1), 1:NZ);

    %% Environmental selection
    Choose  = false(1, N2);
    Zchoose = true(1, NZ);
    while sum(Choose) < K
        Temp = find(Zchoose);
        Jmin = find(rho(Temp) == min(rho(Temp)));
        j    = Temp(Jmin(randi(length(Jmin))));
        I    = find(Choose == 0 & pi(N1 + 1:end) == j);
        if ~isempty(I)
            if rho(j) == 0
                [~, s] = min(d(N1 + I));
            else
                s = randi(length(I));
            end
            Choose(I(s)) = true;
            rho(j) = rho(j) + 1;
        else
            Zchoose(j) = false;
        end
    end
end

function PopCon = extract_constraints(Population)
    consCell = {Population.cons};
    if isempty(consCell) || all(cellfun(@isempty, consCell))
        PopCon = [];
        return;
    end
    maxLen = max(cellfun(@numel, consCell));
    PopSize = numel(Population);
    PopCon = zeros(PopSize, maxLen);
    for i = 1:PopSize
        c = consCell{i};
        if isempty(c)
            continue;
        end
        c = c(:)';
        PopCon(i, 1:numel(c)) = c;
    end
end

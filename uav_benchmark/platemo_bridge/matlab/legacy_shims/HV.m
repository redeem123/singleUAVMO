function score = HV(Population,optimum)
% Small numeric-compatible HV shim for old reference-code algorithms.

    if isa(Population,'INDIVIDUAL')
        PopObj = Population.objs;
    else
        PopObj = double(Population);
    end
    optimum = double(optimum);
    if isempty(PopObj) || isempty(optimum) || size(PopObj,2) ~= size(optimum,2)
        score = 0;
        return;
    end
    [N,M] = size(PopObj);
    fmin = min(min(PopObj,[],1),zeros(1,M));
    fmax = max(optimum,[],1);
    denom = (fmax-fmin)*1.1;
    denom(denom <= 0) = 1;
    PopObj = (PopObj-repmat(fmin,N,1))./repmat(denom,N,1);
    PopObj(any(PopObj>1,2),:) = [];
    RefPoint = ones(1,M);
    if isempty(PopObj)
        score = 0;
    elseif M < 4
        pl = sortrows(PopObj);
        S = {1,pl};
        for k = 1 : M-1
            S_ = {};
            for i = 1 : size(S,1)
                Stemp = Slice(cell2mat(S(i,2)),k,RefPoint);
                for j = 1 : size(Stemp,1)
                    temp(1) = {cell2mat(Stemp(j,1))*cell2mat(S(i,1))};
                    temp(2) = Stemp(j,2);
                    S_ = Add(temp,S_);
                end
            end
            S = S_;
        end
        score = 0;
        for i = 1 : size(S,1)
            p = Head(cell2mat(S(i,2)));
            score = score + cell2mat(S(i,1))*abs(p(M)-RefPoint(M));
        end
    else
        SampleNum = 10000;
        MaxValue = RefPoint;
        MinValue = min(PopObj,[],1);
        Samples = unifrnd(repmat(MinValue,SampleNum,1),repmat(MaxValue,SampleNum,1));
        for i = 1 : size(PopObj,1)
            domi = true(size(Samples,1),1);
            m = 1;
            while m <= M && any(domi)
                domi = domi & PopObj(i,m) <= Samples(:,m);
                m = m + 1;
            end
            Samples(domi,:) = [];
        end
        score = prod(MaxValue-MinValue)*(1-size(Samples,1)/SampleNum);
    end
end

function S = Slice(pl,k,RefPoint)
    p = Head(pl);
    pl = Tail(pl);
    ql = [];
    S = {};
    while ~isempty(pl)
        ql = Insert(p,k+1,ql);
        p_ = Head(pl);
        cell_(1,1) = {abs(p(k)-p_(k))};
        cell_(1,2) = {ql};
        S = Add(cell_,S);
        p = p_;
        pl = Tail(pl);
    end
    ql = Insert(p,k+1,ql);
    cell_(1,1) = {abs(p(k)-RefPoint(k))};
    cell_(1,2) = {ql};
    S = Add(cell_,S);
end

function ql = Insert(p,k,pl)
    flag1 = 0;
    flag2 = 0;
    ql = [];
    hp = Head(pl);
    while ~isempty(pl) && hp(k) < p(k)
        ql = [ql;hp]; %#ok<AGROW>
        pl = Tail(pl);
        hp = Head(pl);
    end
    ql = [ql;p]; %#ok<AGROW>
    m = length(p);
    while ~isempty(pl)
        q = Head(pl);
        for i = k : m
            if p(i) < q(i)
                flag1 = 1;
            else
                if p(i) > q(i)
                    flag2 = 1;
                end
            end
        end
        if ~(flag1 == 1 && flag2 == 0)
            ql = [ql;Head(pl)]; %#ok<AGROW>
        end
        pl = Tail(pl);
    end
end

function p = Head(pl)
    if isempty(pl)
        p = [];
    else
        p = pl(1,:);
    end
end

function ql = Tail(pl)
    if size(pl,1) < 2
        ql = [];
    else
        ql = pl(2:end,:);
    end
end

function S_ = Add(cell_,S)
    n = size(S,1);
    m = 0;
    for k = 1 : n
        if isequal(cell_(2),S(k,2))
            S(k,1) = {cell2mat(S(k,1))+cell2mat(cell_(1))};
            m = 1;
            break;
        end
    end
    if m == 0
        S(n+1,:) = cell_;
    end
    S_ = S;
end

function score = HV(PopObj,optimum)
% Numeric HV for MOEA-2DE fair bridge. Supports four benchmark objectives.

    PopObj = double(PopObj);
    optimum = double(optimum);
    if isempty(PopObj) || size(PopObj,2) ~= size(optimum,2)
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
    if isempty(PopObj)
        score = 0;
        return;
    end
    SampleNum = 10000;
    RefPoint = ones(1,M);
    MinValue = min(PopObj,[],1);
    Samples = unifrnd(repmat(MinValue,SampleNum,1),repmat(RefPoint,SampleNum,1));
    for i = 1 : size(PopObj,1)
        domi = true(size(Samples,1),1);
        m = 1;
        while m <= M && any(domi)
            domi = domi & PopObj(i,m) <= Samples(:,m);
            m = m + 1;
        end
        Samples(domi,:) = [];
    end
    score = prod(RefPoint-MinValue)*(1-size(Samples,1)/SampleNum);
end

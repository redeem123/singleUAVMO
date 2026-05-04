function [Score] = calMetirc(MetricIndex,PopObj,problemIndex) %#ok<INUSD>
% Runtime progress metric compatible with benchmark-objective MOEA-2DE runs.

    PopObj = double(PopObj);
    if isempty(PopObj)
        Score = 0;
        return;
    end
    if MetricIndex == 1
        ref = ones(1,size(PopObj,2));
        Score = HV(PopObj,ref);
    else
        Score = PD(PopObj);
    end
end

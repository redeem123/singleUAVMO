function [Score] = calMetric(MetricIndex, PopObj, problemIndex, M, hvSamples, refPoint)
    % calMetric: Calculate performance metrics
    % MetricIndex: 1 for Hypervolume (HV), 2 for Pure Diversity (PD)
    % M: Number of objectives

    if nargin < 4, M = size(PopObj, 2); end

    if isempty(PopObj)
        Score = 0;
        return;
    end

    if MetricIndex == 1
        PopObj = PopObj(all(isfinite(PopObj), 2), :);
        if isempty(PopObj)
            Score = 0;
            return;
        end
        if nargin < 6 || isempty(refPoint)
            maxVals = max(PopObj, [], 1);
            refPoint = maxVals * 1.1;
            refPoint(refPoint <= 0) = 1;
        end
        if nargin >= 5 && ~isempty(hvSamples)
            Score = HV(PopObj, refPoint, hvSamples);
        else
            Score = HV(PopObj, refPoint);
        end
    else
        Score = PD(PopObj);
    end
end

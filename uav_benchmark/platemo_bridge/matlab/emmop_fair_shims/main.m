function [HV_, PD_, HV_std, PD_std, GlobalBest, runtime_sum, res_S, success_rate, HV_all, PD_all] = main(varargin)
% EMMOP fair-bridge main shim.
% The official main.m re-adds the full reference tree to the top of the
% MATLAB path, which hides benchmark-evaluation shims. This entry point keeps
% the caller's path order intact.

    Global = GLOBAL(varargin{:});
    res_HV = [];
    res_PD = [];
    res_S = [];
    Globals = [];
    HV_all = [];
    PD_all = [];
    failed_times = 0;
    for i = 1:Global.run
        Global.Start();
        res_HV = [res_HV; Global.HV]; %#ok<AGROW>
        res_PD = [res_PD; Global.PD]; %#ok<AGROW>
        Population = Global.result{end};
        Objs = Population.objs;
        if size(Objs,1) > 1
            DistMat = pdist2(Objs, Objs);
            DistMat(DistMat == 0) = inf;
            res_S = [res_S; std(min(DistMat, [], 2))]; %#ok<AGROW>
        else
            res_S = [res_S; 0]; %#ok<AGROW>
        end
        Feasible = find(all(Global.result{end}.cons <= 0, 2), 1);
        if isempty(Feasible)
            failed_times = failed_times + 1;
        end
        Globals = [Globals Global]; %#ok<AGROW>
        Global = GLOBAL(varargin{:});
    end
    HV_all = res_HV;
    PD_all = res_PD;
    HV_ = mean(res_HV, 1);
    PD_ = mean(res_PD, 1);
    HV_std = std(res_HV, 0, 1);
    PD_std = std(res_PD, 0, 1);
    if isempty(res_HV)
        GlobalBest = Globals(1);
    else
        [~, idx] = max(res_HV(:, end));
        GlobalBest = Globals(idx);
    end
    runtime_sum = sum([Globals.runtime]);
    success_rate = 1 - failed_times / max(1,length(Globals));
end

function [bestScores, gen_hv] = run_momfea(model, params)
% run_momfea - Run PlatEMO MO-MFEA on UAV benchmark using multitask adapter.
    [bestScores, gen_hv] = run_momfea_core(model, params, 'MOMFEA');
end

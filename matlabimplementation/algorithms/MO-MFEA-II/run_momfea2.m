function [bestScores, gen_hv] = run_momfea2(model, params)
% run_momfea2 - Run PlatEMO MO-MFEA-II on UAV benchmark using multitask adapter.
    [bestScores, gen_hv] = run_momfea_core(model, params, 'MOMFEAII');
end

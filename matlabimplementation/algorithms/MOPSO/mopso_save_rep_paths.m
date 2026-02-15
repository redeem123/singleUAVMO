function mopso_save_rep_paths(rep, model, run_dir)
    for i_p = 1:numel(rep)
        local_dt_sv = struct();
        pos = rep(i_p).Position;
        x_full = [model.start(1), pos.x, model.end(1)]';
        y_full = [model.start(2), pos.y, model.end(2)]';
        start_z = model.start(3);
        end_z = model.end(3);
        if isfield(model, 'safeH') && ~isempty(model.safeH)
            start_z = model.safeH;
            end_z = model.safeH;
        end
        z_rel = [start_z, pos.z, end_z]';
        z_abs_path = zeros(size(x_full));
        for k = 1:length(x_full)
            xi = max(1, min(model.xmax, round(x_full(k))));
            yi = max(1, min(model.ymax, round(y_full(k))));
            z_abs_path(k) = z_rel(k) + model.H(yi, xi);
        end
        local_dt_sv.path = [x_full, y_full, z_abs_path];
        local_dt_sv.objs = rep(i_p).Cost';
        mopso_save_data(fullfile(run_dir, sprintf('bp_%d.mat', i_p)), local_dt_sv);
    end
end

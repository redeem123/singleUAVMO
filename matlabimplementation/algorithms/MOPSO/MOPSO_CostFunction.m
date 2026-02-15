function cost = MOPSO_CostFunction(sol, model, ~)
    J_large = inf;

    xs = model.start(1); ys = model.start(2); zs = model.start(3);
    xf = model.end(1); yf = model.end(2); zf = model.end(3);
    if isfield(model, 'safeH') && ~isempty(model.safeH)
        zs = model.safeH;
        zf = model.safeH;
    end

    x_all = [xs, sol.x, xf];
    y_all = [ys, sol.y, yf];
    z_rel = [zs, sol.z, zf];

    N = size(x_all, 2);
    path = zeros(N, 3);

    collision = false;
    for i = 1:N
        xi = max(1, min(model.xmax, round(x_all(i))));
        yi = max(1, min(model.ymax, round(y_all(i))));

        abs_z = z_rel(i) + model.H(yi, xi);

        if z_rel(i) < 0
            collision = true;
        end
        path(i, :) = [x_all(i), y_all(i), abs_z];
    end

    if collision
        cost = [J_large; J_large; J_large; J_large];
        return;
    end

    cost = evaluate_path(path, model)';
end

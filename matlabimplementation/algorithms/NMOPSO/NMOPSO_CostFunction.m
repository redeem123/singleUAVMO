function cost = NMOPSO_CostFunction(sol, model, ~)
    % NMOPSO_CostFunction: Reconstructs path and delegates evaluation
    J_large = inf;
    
    % Reconstruct absolute path
    xs = model.start(1); ys = model.start(2); zs = model.start(3);
    xf = model.end(1); yf = model.end(2); zf = model.end(3);
    if isfield(model, 'safeH') && ~isempty(model.safeH)
        zs = model.safeH;
        zf = model.safeH;
    end
    
    x_all = [xs, sol.x, xf];
    y_all = [ys, sol.y, yf];
    z_rel = [zs, sol.z, zf]; % Relative or absolute depending on SphericalToCart
    
    N = size(x_all, 2); 
    path = zeros(N, 3);
    
    collision = false;
    for i = 1:N
        xi = max(1, min(model.xmax, round(x_all(i))));
        yi = max(1, min(model.ymax, round(y_all(i))));
        
        % In NMOPSO, sol.z was treated as relative height above ground
        abs_z = z_rel(i) + model.H(yi, xi);
        
        % Check for crashing into ground
        if z_rel(i) < 0
            collision = true;
        end
        path(i, :) = [x_all(i), y_all(i), abs_z];
    end
    
    if collision
        cost = [J_large; J_large; J_large; J_large];
        return;
    end
    
    % Delegate to unified master evaluator
    % Transpose to [1 x 4] to match NMOPSO expectations
    cost = evaluate_path(path, model)';
end

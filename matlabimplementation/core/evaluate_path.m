function objs = evaluate_path(path, model)
    % evaluate_path: Objective evaluator aligned with NMOPSO paper (F1-F4).
    % path: [N x 3] absolute coordinates (x, y, z)
    % model: terrainStruct containing H, nofly_c, nofly_r, etc.

    J_inf = inf;

    if isempty(path) || size(path, 2) ~= 3
        objs = [J_inf, J_inf, J_inf, J_inf];
        return;
    end

    x = path(:, 1);
    y = path(:, 2);
    z_abs = path(:, 3);

    if any(x < model.xmin | x > model.xmax | y < model.ymin | y > model.ymax)
        objs = [J_inf, J_inf, J_inf, J_inf];
        return;
    end

    xi = max(1, min(model.xmax, round(x)));
    yi = max(1, min(model.ymax, round(y)));
    groundH = model.H(sub2ind(size(model.H), yi, xi));
    z_rel = z_abs - groundH;

    % F1: Path length
    start_pt = path(1, :);
    end_pt = path(end, :);

    Rmin = 0;
    if isfield(model, 'rmin') && ~isempty(model.rmin)
        Rmin = double(model.rmin);
    elseif isfield(model, 'n') && ~isempty(model.n) && double(model.n) > 0
        path_diag = norm(end_pt - start_pt);
        Rmin = path_diag / (3 * double(model.n));
    end

    seg_vec = diff(path, 1, 1);
    seg_len = sqrt(sum(seg_vec.^2, 2));
    if any(seg_len <= Rmin)
        F1 = J_inf;
    else
        total_len = sum(seg_len);
        if total_len > 0
            F1 = 1 - norm(end_pt - start_pt) / total_len;
        else
            F1 = J_inf;
        end
    end

    % F2: Clearance from terrain and obstacles
    obstacles = [];
    if isfield(model, 'threats') && ~isempty(model.threats)
        obstacles = model.threats;
    end
    if isfield(model, 'nofly_c') && isfield(model, 'nofly_r') && ~isempty(model.nofly_c)
        c = double(model.nofly_c);
        r = double(model.nofly_r);
        if size(c, 2) ~= 2
            c = c(:)';
        end
        if numel(r) == 1
            r = repmat(r, size(c, 1), 1);
        end
        nf = [c(:, 1), c(:, 2), zeros(size(c, 1), 1), r(:)];
        obstacles = [obstacles; nf];
    end

    D = 1;
    S = 10;
    if isfield(model, 'droneSize') && ~isempty(model.droneSize)
        D = double(model.droneSize);
    elseif isfield(model, 'drone_size') && ~isempty(model.drone_size)
        D = double(model.drone_size);
    end
    if isfield(model, 'safeDist') && ~isempty(model.safeDist)
        S = double(model.safeDist);
    elseif isfield(model, 'safe_dist') && ~isempty(model.safe_dist)
        S = double(model.safe_dist);
    end

    step_size = 1;
    if isfield(model, 'collisionStep') && isnumeric(model.collisionStep) && isfinite(model.collisionStep)
        step_size = model.collisionStep;
    end
    if step_size <= 0
        step_size = 1;
    end

    full_path = interpolate_path(path, step_size);
    xf = full_path(:, 1);
    yf = full_path(:, 2);
    zf_abs = full_path(:, 3);
    xif = max(1, min(model.xmax, round(xf)));
    yif = max(1, min(model.ymax, round(yf)));
    groundHf = model.H(sub2ind(size(model.H), yif, xif));
    zf_rel = zf_abs - groundHf;

    n = size(full_path, 1);
    if n < 2
        F2 = 0;
    else
        K = size(obstacles, 1);
        Tk_sum = 0;
        for j = 1:n-1
            ground_clear = min(zf_rel(j), zf_rel(j+1));
            min_clear = ground_clear;
            if K > 0
                p1 = [xf(j), yf(j)];
                p2 = [xf(j+1), yf(j+1)];
                for k = 1:K
                    ck = obstacles(k, 1:2);
                    Rk = obstacles(k, 4);
                    dk = dist_point_to_segment_2d(ck, p1, p2) - Rk;
                    if dk < min_clear
                        min_clear = dk;
                    end
                end
            end

            if min_clear >= D + S
                Tk = 0;
            elseif min_clear > D
                Tk = 1 - (min_clear - D) / S;
            else
                Tk = J_inf;
            end

            if isinf(Tk)
                Tk_sum = J_inf;
                break;
            end
            Tk_sum = Tk_sum + Tk;
        end

        if isinf(Tk_sum)
            F2 = J_inf;
        else
            F2 = Tk_sum / (n - 1);
        end
    end

    % F3: Flight altitude (relative to ground)
    hmax = double(model.zmax);
    hmin = double(model.zmin);
    if hmax <= hmin
        F3 = J_inf;
    else
        hmean = (hmax + hmin) / 2;
        Hij = zeros(size(z_rel));
        for j = 1:numel(z_rel)
            hij = z_rel(j);
            if hij < hmin || hij > hmax
                Hij(j) = J_inf;
            else
                Hij(j) = 2 * abs(hij - hmean) / (hmax - hmin);
            end
        end
        if any(isinf(Hij))
            F3 = J_inf;
        else
            F3 = mean(Hij);
        end
    end

    % F4: Smoothness
    n = size(path, 1);
    if n < 3
        F4 = 0;
    else
        angles = zeros(n - 2, 1);
        for j = 1:n-2
            v1 = path(j+1, :) - path(j, :);
            v2 = path(j+2, :) - path(j+1, :);
            if norm(v1) == 0 || norm(v2) == 0
                angles(j) = 0;
            else
                angles(j) = atan2(norm(cross(v1, v2)), dot(v1, v2));
            end
        end
        F4 = mean(abs(angles) / pi);
    end

    objs = [F1, F2, F3, F4];
end

function dist = dist_point_to_segment_2d(p, a, b)
    % p, a, b are 1x2 vectors
    ab = b - a;
    ap = p - a;
    ab2 = dot(ab, ab);
    if ab2 == 0
        dist = norm(ap);
        return;
    end
    t = dot(ap, ab) / ab2;
    t = max(0, min(1, t));
    proj = a + t * ab;
    dist = norm(p - proj);
end

function new_path = interpolate_path(path, step_size)
    n_points = size(path, 1);
    n_segments = n_points - 1;
    steps_per_segment = zeros(n_segments, 1);
    for i = 1:n_segments
        dist = norm(path(i+1, :) - path(i, :));
        steps_per_segment(i) = max(1, ceil(dist / step_size));
    end

    total_points = 1 + sum(steps_per_segment);
    new_path = zeros(total_points, 3);
    idx = 1;
    new_path(idx, :) = path(1, :);
    for i = 1:n_segments
        p1 = path(i, :);
        p2 = path(i+1, :);
        steps = steps_per_segment(i);
        for s = 1:steps
            t = s / steps;
            idx = idx + 1;
            new_path(idx, :) = (1 - t) * p1 + t * p2;
        end
    end
end

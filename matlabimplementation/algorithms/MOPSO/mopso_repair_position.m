function pos = mopso_repair_position(pos, model, VarMin, VarMax)
    % Clamp to bounds and enforce a forward progression in XY.

    zLower = VarMin.z;
    if ~isempty(zLower)
        zLower = max(0, zLower);
    else
        zLower = 0;
    end

    pos.x = max(VarMin.x, min(VarMax.x, pos.x));
    pos.y = max(VarMin.y, min(VarMax.y, pos.y));
    pos.z = max(zLower, min(VarMax.z, pos.z));

    dx = model.end(1) - model.start(1);
    dy = model.end(2) - model.start(2);
    denom = dx*dx + dy*dy;
    if denom > 0
        t = ((pos.x - model.start(1)) * dx + (pos.y - model.start(2)) * dy) / denom;
        [~, idx] = sort(t);
        pos.x = pos.x(idx);
        pos.y = pos.y(idx);
        pos.z = pos.z(idx);
    end

    minStep = 0;
    if isfield(model, 'rmin') && ~isempty(model.rmin)
        minStep = double(model.rmin);
    elseif isfield(model, 'n') && ~isempty(model.n) && double(model.n) > 0
        path_diag = norm(model.end - model.start);
        minStep = path_diag / (3 * double(model.n));
    end

    if minStep > 0 && denom > 0
        dir_xy = [dx, dy] / sqrt(denom);
        for i = 2:numel(pos.x)
            prev = [pos.x(i-1), pos.y(i-1)];
            cur = [pos.x(i), pos.y(i)];
            if norm(cur - prev) < minStep
                cur = prev + dir_xy * minStep;
                pos.x(i) = max(VarMin.x, min(VarMax.x, cur(1)));
                pos.y(i) = max(VarMin.y, min(VarMax.y, cur(2)));
            end
        end
    end
end

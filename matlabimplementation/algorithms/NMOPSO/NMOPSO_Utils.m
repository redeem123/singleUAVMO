classdef NMOPSO_Utils
    methods(Static)
        
        function b = Dominates(x,y)
            if isstruct(x)
                x = x.Cost;
            end
            if isstruct(y)
                y = y.Cost;
            end
            % Minimization: x dominates y if x is better in all and strictly better in one
            % Ensure finite costs to be considered valid for dominance
            b = all(x <= y) && any(x < y) && all(isfinite(x));
        end

        function pop = DetermineDomination(pop)
            nPop = numel(pop);
            for i = 1:nPop
                pop(i).IsDominated = false;
            end
            % Compare every pair
            for i = 1:nPop
                for j = 1:nPop
                    if i == j, continue; end
                    if NMOPSO_Utils.Dominates(pop(j), pop(i))
                        pop(i).IsDominated = true;
                        break;
                    end
                end
            end
        end

        function Grid = CreateGrid(pop, nGrid, alpha)
            c = [pop.Cost]; 
            cmin = min(c, [], 2); 
            cmax = max(c, [], 2);
            dc = cmax-cmin;
            cmin = cmin-alpha*dc;
            cmax = cmax+alpha*dc;
            nObj = size(c, 1); 
            empty_grid.LB = [];
            empty_grid.UB = [];
            Grid = repmat(empty_grid, nObj, 1);
            for j = 1:nObj
                cj = linspace(cmin(j), cmax(j), nGrid+1); 
                Grid(j).LB = [-inf, cj];
                Grid(j).UB = [cj, +inf];
            end
        end

        function particle = FindGridIndex(particle, Grid)
            nObj = numel(particle.Cost);
            nGrid = numel(Grid(1).LB)-2;
            particle.GridSubIndex = zeros(1, nObj);
            idx = zeros(1, nObj);
            for j = 1:nObj
                matches = find(particle.Cost(j) < Grid(j).UB, 1, 'first');
                if isempty(matches)
                    idx(j) = nGrid;
                else
                    idx(j) = matches;
                end
            end
            particle.GridSubIndex = idx;
            particle.GridIndex = particle.GridSubIndex(1);
            for j = 2:nObj
                particle.GridIndex = (particle.GridIndex-1) * nGrid + particle.GridSubIndex(j);
            end
        end

        function rep = DeleteOneRepMember(rep, gamma)
            GI = [rep.GridIndex];
            OC = unique(GI);
            N = zeros(size(OC));
            for k = 1:numel(OC)
                N(k) = sum(GI == OC(k));
            end
            P = exp(gamma*N);
            P = P/sum(P);
            sci = NMOPSO_Utils.RouletteWheelSelection(P);
            sc = OC(sci);
            SCM = find(GI == sc);
            sm = SCM(randi([1 numel(SCM)]));
            rep(sm) = [];
        end

        function i = RouletteWheelSelection(P)
            r = rand;
            C = cumsum(P); 
            i = find(r <= C, 1, 'first');
        end

        function leader = SelectLeader(rep, beta)
            GI = [rep.GridIndex];
            OC = unique(GI);
            N = zeros(size(OC));
            for k = 1:numel(OC)
                N(k) = sum(GI == OC(k));
            end
            P = exp(-beta*N);
            P = P/sum(P);
            sci = NMOPSO_Utils.RouletteWheelSelection(P);
            sc = OC(sci);
            SCM = find(GI == sc);
            sm = SCM(randi([1 numel(SCM)]));
            leader = rep(sm);
        end

        function idx = SelectLeaderRef(rep, Z, nSelect)
            if nargin < 3 || isempty(nSelect)
                nSelect = 1;
            end
            if isempty(rep)
                idx = [];
                return;
            end
            PopObj = horzcat(rep.Cost)';
            if size(PopObj, 2) ~= numel(rep(1).Cost)
                PopObj = PopObj';
            end
            PopObj = NMOPSO_Utils.NormalizeObjectives(PopObj);
            Z = NMOPSO_Utils.NormalizeObjectives(Z);

            Cosine = PopObj * Z';
            [~, pi] = max(Cosine, [], 2);
            NZ = size(Z, 1);
            rho = hist(pi, 1:NZ);

            idx = zeros(1, nSelect);
            Zchoose = true(1, NZ);
            count = 0;
            while count < nSelect
                Temp = find(Zchoose);
                if isempty(Temp)
                    idx(count+1:end) = randi(numel(rep), 1, nSelect - count);
                    break;
                end
                Jmin = find(rho(Temp) == min(rho(Temp)));
                j = Temp(Jmin(randi(length(Jmin))));
                I = find(pi == j);
                if ~isempty(I)
                    count = count + 1;
                    idx(count) = I(randi(length(I)));
                    rho(j) = rho(j) + 1;
                else
                    Zchoose(j) = false;
                end
            end
        end

        function atlasCfg = BuildAtlasConfig(ablation)
            atlasCfg = struct();
            atlasCfg.enabled = false;
            atlasCfg.nTopologyBins = 24;
            atlasCfg.nRobustBins = 4;
            atlasCfg.maxObstacles = 3;
            atlasCfg.hashLevels = 6;
            atlasCfg.objectiveWeight = 0.5;
            atlasCfg.atlasWeight = 0.5;

            if nargin < 1 || isempty(ablation)
                return;
            end

            if isfield(ablation, 'useTopologyRobustArchive')
                atlasCfg.enabled = logical(ablation.useTopologyRobustArchive);
            end
            if isfield(ablation, 'atlasTopologyBins')
                atlasCfg.nTopologyBins = max(2, round(double(ablation.atlasTopologyBins)));
            end
            if isfield(ablation, 'atlasRobustBins')
                atlasCfg.nRobustBins = max(2, round(double(ablation.atlasRobustBins)));
            end
            if isfield(ablation, 'atlasMaxObstacles')
                atlasCfg.maxObstacles = max(1, round(double(ablation.atlasMaxObstacles)));
            end
            if isfield(ablation, 'atlasHashLevels')
                atlasCfg.hashLevels = max(2, round(double(ablation.atlasHashLevels)));
            end
            if isfield(ablation, 'atlasObjectiveWeight')
                atlasCfg.objectiveWeight = max(0, double(ablation.atlasObjectiveWeight));
            end
            if isfield(ablation, 'atlasTopologyWeight')
                atlasCfg.atlasWeight = max(0, double(ablation.atlasTopologyWeight));
            end

            wsum = atlasCfg.objectiveWeight + atlasCfg.atlasWeight;
            if wsum <= 0
                atlasCfg.objectiveWeight = 0.5;
                atlasCfg.atlasWeight = 0.5;
            else
                atlasCfg.objectiveWeight = atlasCfg.objectiveWeight / wsum;
                atlasCfg.atlasWeight = atlasCfg.atlasWeight / wsum;
            end
        end

        function particle = UpdateAtlasMetadata(particle, model, representation, atlasCfg, cartPos)
            if nargin < 4 || isempty(atlasCfg)
                atlasCfg = NMOPSO_Utils.BuildAtlasConfig([]);
            end
            if nargin < 5 || isempty(cartPos)
                cartPos = NMOPSO_Utils.PositionToCart(particle.Position, model, representation);
            end

            path = NMOPSO_Utils.CartToAbsolutePath(cartPos, model);
            sig = NMOPSO_Utils.TopologySignature(path, model, atlasCfg.maxObstacles);
            topoBin = NMOPSO_Utils.TopologyBinFromSignature(sig, atlasCfg);
            [robScore, robBin] = NMOPSO_Utils.RobustnessFromCost(particle.Cost, atlasCfg.nRobustBins);

            particle.TopologySignature = sig;
            particle.TopologyBin = topoBin;
            particle.RobustnessScore = robScore;
            particle.RobustnessBin = robBin;
            particle.AtlasCellIndex = (topoBin - 1) * atlasCfg.nRobustBins + robBin;
        end

        function pop = RefreshAtlasCellIndex(pop, atlasCfg)
            if isempty(pop)
                return;
            end
            if nargin < 2 || isempty(atlasCfg)
                atlasCfg = NMOPSO_Utils.BuildAtlasConfig([]);
            end

            for i = 1:numel(pop)
                topoBin = 1;
                if isfield(pop, 'TopologyBin') && ~isempty(pop(i).TopologyBin) && ...
                        isfinite(pop(i).TopologyBin)
                    topoBin = max(1, min(atlasCfg.nTopologyBins, round(double(pop(i).TopologyBin))));
                elseif isfield(pop, 'TopologySignature') && ~isempty(pop(i).TopologySignature)
                    topoBin = NMOPSO_Utils.TopologyBinFromSignature(pop(i).TopologySignature, atlasCfg);
                    pop(i).TopologyBin = topoBin;
                end

                [robScore, robBin] = NMOPSO_Utils.RobustnessFromCost(pop(i).Cost, atlasCfg.nRobustBins);
                pop(i).RobustnessScore = robScore;
                pop(i).RobustnessBin = robBin;
                pop(i).AtlasCellIndex = (topoBin - 1) * atlasCfg.nRobustBins + robBin;
            end
        end

        function rep = DeleteOneRepMemberAtlas(rep, gamma, objectiveWeight, atlasWeight)
            if isempty(rep)
                return;
            end
            if nargin < 3 || isempty(objectiveWeight)
                objectiveWeight = 0.5;
            end
            if nargin < 4 || isempty(atlasWeight)
                atlasWeight = 0.5;
            end

            [objOcc, atlasOcc] = NMOPSO_Utils.ArchiveOccupancies(rep);
            occ = objectiveWeight * objOcc + atlasWeight * atlasOcc;
            P = exp(gamma * occ);
            if ~all(isfinite(P)) || sum(P) <= 0
                P = ones(1, numel(rep)) / numel(rep);
            else
                P = P / sum(P);
            end

            sm = NMOPSO_Utils.RouletteWheelSelection(P);
            if isempty(sm)
                sm = randi(numel(rep));
            end
            rep(sm) = [];
        end

        function leader = SelectLeaderAtlas(rep, beta, objectiveWeight, atlasWeight)
            if isempty(rep)
                leader = [];
                return;
            end
            if nargin < 3 || isempty(objectiveWeight)
                objectiveWeight = 0.5;
            end
            if nargin < 4 || isempty(atlasWeight)
                atlasWeight = 0.5;
            end

            [objOcc, atlasOcc] = NMOPSO_Utils.ArchiveOccupancies(rep);
            occ = objectiveWeight * objOcc + atlasWeight * atlasOcc;
            P = exp(-beta * occ);
            if ~all(isfinite(P)) || sum(P) <= 0
                P = ones(1, numel(rep)) / numel(rep);
            else
                P = P / sum(P);
            end

            sm = NMOPSO_Utils.RouletteWheelSelection(P);
            if isempty(sm)
                sm = randi(numel(rep));
            end
            leader = rep(sm);
        end

        function [objOcc, atlasOcc] = ArchiveOccupancies(rep)
            n = numel(rep);
            objOcc = ones(1, n);
            atlasOcc = ones(1, n);
            if isempty(rep)
                return;
            end

            if isfield(rep, 'GridIndex')
                GI = [rep.GridIndex];
                if numel(GI) == n && all(isfinite(GI))
                    [~, ~, ic] = unique(GI);
                    counts = accumarray(ic(:), 1);
                    objOcc = counts(ic)';
                end
            end

            if isfield(rep, 'AtlasCellIndex')
                AI = [rep.AtlasCellIndex];
                if numel(AI) == n && all(isfinite(AI))
                    [~, ~, ic] = unique(AI);
                    counts = accumarray(ic(:), 1);
                    atlasOcc = counts(ic)';
                end
            end
        end

        function nrt = ArchiveRegionCount(pm)
            nrt = 1;
            if isempty(pm) || ~isstruct(pm)
                return;
            end

            key = [];
            if isfield(pm, 'AtlasCellIndex')
                key = [pm.AtlasCellIndex];
                key = key(isfinite(key));
            end
            if isempty(key) && isfield(pm, 'GridIndex')
                key = [pm.GridIndex];
                key = key(isfinite(key));
            end

            if ~isempty(key)
                nrt = numel(unique(key));
                if nrt < 1
                    nrt = 1;
                end
            end
        end

        function [score, bin] = RobustnessFromCost(cost, nBins)
            if nargin < 2 || isempty(nBins)
                nBins = 4;
            end
            nBins = max(2, round(double(nBins)));

            score = 0;
            if isempty(cost) || numel(cost) < 4
                bin = 1;
                return;
            end

            f2 = double(cost(2));
            f4 = double(cost(4));
            if ~isfinite(f2)
                score = 0;
            else
                smoothPenalty = 0;
                if isfinite(f4) && f4 > 0
                    smoothPenalty = 0.35 * f4;
                end
                score = 1 / (1 + max(0, f2) + smoothPenalty);
            end
            score = max(0, min(1, score));
            bin = min(nBins, max(1, floor(score * nBins) + 1));
        end

        function path = CartToAbsolutePath(cart, model)
            xs = model.start(1); ys = model.start(2); zs = model.start(3);
            xf = model.end(1); yf = model.end(2); zf = model.end(3);
            if isfield(model, 'safeH') && ~isempty(model.safeH)
                zs = model.safeH;
                zf = model.safeH;
            end

            x_all = [xs, cart.x, xf];
            y_all = [ys, cart.y, yf];
            z_rel = [zs, cart.z, zf];

            nPts = numel(x_all);
            path = zeros(nPts, 3);
            for i = 1:nPts
                xi = max(1, min(model.xmax, round(x_all(i))));
                yi = max(1, min(model.ymax, round(y_all(i))));
                ground = 0;
                if isfield(model, 'H') && ~isempty(model.H)
                    ground = model.H(yi, xi);
                end
                path(i, :) = [x_all(i), y_all(i), z_rel(i) + ground];
            end
        end

        function sig = TopologySignature(path, model, maxObs)
            if nargin < 3 || isempty(maxObs)
                maxObs = 3;
            end
            maxObs = max(1, round(double(maxObs)));
            sig = zeros(1, 4 + 3 * maxObs);
            if isempty(path) || size(path, 2) < 2 || size(path, 1) < 2
                return;
            end

            xy = double(path(:, 1:2));
            dx = max(1, double(model.xmax) - double(model.xmin));
            dy = max(1, double(model.ymax) - double(model.ymin));
            mapDiag = sqrt(dx^2 + dy^2);

            dxy = diff(xy, 1, 1);
            segLen = sqrt(sum(dxy.^2, 2));
            pathLenNorm = sum(segLen) / mapDiag;

            heading = atan2(dxy(:, 2), dxy(:, 1));
            if numel(heading) >= 2
                turn = NMOPSO_Utils.WrapToPi(diff(heading));
                meanTurn = mean(abs(turn)) / pi;
                signedTurn = sum(turn) / (pi * max(1, numel(turn)));
                turnStd = std(turn) / pi;
            else
                meanTurn = 0;
                signedTurn = 0;
                turnStd = 0;
            end
            sig(1:4) = [pathLenNorm, meanTurn, signedTurn, turnStd];

            obsFeat = zeros(1, maxObs * 3);
            [centers, radii] = NMOPSO_Utils.ExtractObstacles(model, maxObs);
            if ~isempty(centers)
                baseDir = xy(end, :) - xy(1, :);
                if norm(baseDir) < 1e-12
                    baseDir = [1, 0];
                end

                for k = 1:size(centers, 1)
                    c = centers(k, :);
                    r = radii(k);
                    dist = sqrt((xy(:, 1) - c(1)).^2 + (xy(:, 2) - c(2)).^2);
                    [minDist, idx] = min(dist);

                    sideVec = xy(idx, :) - c;
                    side = sign(baseDir(1) * sideVec(2) - baseDir(2) * sideVec(1));
                    if ~isfinite(side)
                        side = 0;
                    end

                    ang = unwrap(atan2(xy(:, 2) - c(2), xy(:, 1) - c(1)));
                    winding = (ang(end) - ang(1)) / (2 * pi);
                    clearance = (minDist - r) / mapDiag;

                    base = 3 * (k - 1);
                    obsFeat(base + 1:base + 3) = [side, winding, clearance];
                end
            end
            sig(5:end) = obsFeat;
            sig(~isfinite(sig)) = 0;
        end

        function bin = TopologyBinFromSignature(sig, atlasCfg)
            if isempty(sig)
                bin = 1;
                return;
            end
            nBins = max(2, round(double(atlasCfg.nTopologyBins)));
            levels = max(2, round(double(atlasCfg.hashLevels)));
            normSig = NMOPSO_Utils.NormalizeSignatureForHash(sig);
            q = floor(normSig * levels);
            q = max(0, min(levels - 1, q));

            h = 0;
            for i = 1:numel(q)
                primeLike = 2 * i + 1;
                h = mod(h + (q(i) + 1) * primeLike, nBins);
            end
            bin = h + 1;
        end

        function x = NormalizeSignatureForHash(sig)
            x = zeros(size(sig));
            if isempty(sig)
                return;
            end

            if numel(sig) >= 1
                x(1) = min(max(sig(1), 0), 3) / 3;
            end
            if numel(sig) >= 2
                x(2) = min(max(sig(2), 0), 1);
            end
            if numel(sig) >= 3
                x(3) = (min(max(sig(3), -1), 1) + 1) / 2;
            end
            if numel(sig) >= 4
                x(4) = min(max(sig(4), 0), 1);
            end

            for i = 5:numel(sig)
                localIdx = mod(i - 5, 3) + 1;
                if localIdx == 1
                    x(i) = (min(max(sig(i), -1), 1) + 1) / 2;
                elseif localIdx == 2
                    x(i) = (min(max(sig(i), -1), 1) + 1) / 2;
                else
                    x(i) = (min(max(sig(i), -0.2), 0.2) + 0.2) / 0.4;
                end
            end
            x(~isfinite(x)) = 0;
        end

        function wrapped = WrapToPi(theta)
            wrapped = mod(theta + pi, 2 * pi) - pi;
        end

        function [centers, radii] = ExtractObstacles(model, maxObs)
            centers = [];
            radii = [];
            if nargin < 2 || isempty(maxObs)
                maxObs = inf;
            end

            if isfield(model, 'nofly_c') && isfield(model, 'nofly_r') && ~isempty(model.nofly_c)
                c = double(model.nofly_c);
                if isvector(c) && numel(c) == 2
                    c = reshape(c, 1, 2);
                end
                if size(c, 2) > 2
                    c = c(:, 1:2);
                end

                r = double(model.nofly_r);
                if isempty(r)
                    r = 0;
                end
                if isscalar(r)
                    r = repmat(r, size(c, 1), 1);
                else
                    r = r(:);
                    if numel(r) < size(c, 1)
                        r(end+1:size(c, 1)) = r(end);
                    end
                end

                centers = [centers; c];
                radii = [radii; r(1:size(c, 1))];
            end

            if isfield(model, 'threats') && ~isempty(model.threats)
                th = double(model.threats);
                if size(th, 2) >= 4
                    centers = [centers; th(:, 1:2)];
                    radii = [radii; th(:, 4)];
                end
            end

            if isempty(centers)
                return;
            end

            valid = all(isfinite(centers), 2) & isfinite(radii) & radii > 0;
            centers = centers(valid, :);
            radii = radii(valid);
            if isempty(centers)
                return;
            end

            [~, ord] = sort(radii, 'descend');
            k = min(double(maxObs), numel(ord));
            ord = ord(1:k);
            centers = centers(ord, :);
            radii = radii(ord);
        end
        
        function xnew = Mutate(x, pm, delta, VarMax, VarMin, representation)
            if nargin < 6 || isempty(representation)
                representation = '';
            end
            if isnumeric(representation)
                if representation == 0
                    representation = 'CC';
                else
                    representation = 'SC';
                end
            end
            if isstring(representation)
                representation = char(representation);
            end
            representation = upper(strtrim(representation));

            if strcmp(representation, 'CC')
                nVar = numel(x.Position.x);
                nrt = NMOPSO_Utils.ArchiveRegionCount(pm);
                beta = tanh(delta / nrt);

                stepX = (VarMax.x - VarMin.x) * beta;
                stepY = (VarMax.y - VarMin.y) * beta;
                stepZ = (VarMax.z - VarMin.z) * beta;

                xnew.x = x.Position.x + randn(1, nVar) .* stepX;
                xnew.y = x.Position.y + randn(1, nVar) .* stepY;
                xnew.z = x.Position.z + randn(1, nVar) .* stepZ;

                xnew.x = min(max(xnew.x, VarMin.x), VarMax.x);
                xnew.y = min(max(xnew.y, VarMin.y), VarMax.y);
                xnew.z = min(max(xnew.z, VarMin.z), VarMax.z);
                return;
            end

            nVar = numel(x.Position.r);
            nrt = NMOPSO_Utils.ArchiveRegionCount(pm);
            beta = tanh(delta / nrt);
            
            xnew.r = x.Position.r + randn(1, nVar) .* x.Best.Position.r * beta;
            xnew.phi = x.Position.phi + randn(1, nVar) .* x.Best.Position.phi * beta;
            xnew.psi = x.Position.psi + randn(1, nVar) .* x.Best.Position.psi * beta;

            xnew.r = min(max(xnew.r, VarMin.r), VarMax.r);
            xnew.phi = min(max(xnew.phi, VarMin.phi), VarMax.phi);
            xnew.psi = min(max(xnew.psi, VarMin.psi), VarMax.psi);
        end

        function X = NormalizeObjectives(X)
            if isempty(X)
                return;
            end
            minVals = min(X, [], 1);
            maxVals = max(X, [], 1);
            rangeVals = maxVals - minVals;
            rangeVals(rangeVals <= 0) = 1;
            X = (X - minVals) ./ rangeVals;
            norms = sqrt(sum(X.^2, 2));
            norms(norms == 0) = 1;
            X = X ./ norms;
        end

        function cart = PositionToCart(position, model, representation)
            if nargin < 3 || isempty(representation)
                if isfield(position, 'r') && isfield(position, 'phi') && isfield(position, 'psi')
                    representation = 'SC';
                else
                    representation = 'CC';
                end
            end
            if isnumeric(representation)
                if representation == 0
                    representation = 'CC';
                else
                    representation = 'SC';
                end
            end
            if isstring(representation)
                representation = char(representation);
            end
            representation = upper(strtrim(representation));

            if strcmp(representation, 'SC')
                cart = NMOPSO_Utils.SphericalToCart(position, model);
                return;
            end

            cart = struct();
            cart.x = position.x;
            cart.y = position.y;
            cart.z = position.z;
            cart.x = max(model.xmin, min(model.xmax, cart.x));
            cart.y = max(model.ymin, min(model.ymax, cart.y));
            cart.z = max(model.zmin, min(model.zmax, cart.z));
        end
        
        function A = TransfomationMatrix(r, phi, psi)
            % Returns a 4x4 transformation matrix
            cp = cos(phi); sp = sin(phi);
            cs = cos(-psi); ss = sin(-psi);
            
            Rot_z = [ cp, -sp, 0, 0; sp,  cp, 0, 0; 0, 0, 1, 0; 0, 0, 0, 1];
            Rot_y = [ cs, 0, ss, 0; 0, 1, 0, 0; -ss, 0, cs, 0; 0, 0, 0, 1];
            Trans_x = [1 0 0 r; 0 1 0 0; 0 0 1 0; 0 0 0 1];
            A = Rot_z * Rot_y * Trans_x;
        end

        function position = SphericalToCart(solution, model)
            % Optimized Spherical to Cartesian conversion
            % solution has fields r, phi, psi (1 x n)
            
            n = length(solution.r);
            xs = model.start(1); ys = model.start(2); zs = model.start(3);
            xf = model.end(1); yf = model.end(2); zf = model.end(3);
            if isfield(model, 'safeH') && ~isempty(model.safeH)
                zs = model.safeH;
                zf = model.safeH;
            end
            
            % Base start position and initial orientation towards target
            dirVector = [xf - xs; yf - ys; zf - zs];
            phistart = atan2(dirVector(2), dirVector(1));
            psistart = atan2(dirVector(3), norm(dirVector(1:2)));
            
            % Global start frame
            startMat = [1, 0, 0, xs; 0, 1, 0, ys; 0, 0, 1, zs; 0, 0, 0, 1];
            initialDir = NMOPSO_Utils.TransfomationMatrix(0, phistart, psistart);
            currentPos = startMat * initialDir;
            
            x = zeros(1, n); y = zeros(1, n); z = zeros(1, n);
            
            % Recursively apply transformations
            for i = 1:n
                T = NMOPSO_Utils.TransfomationMatrix(solution.r(i), solution.phi(i), solution.psi(i));
                currentPos = currentPos * T;
                x(i) = currentPos(1,4);
                y(i) = currentPos(2,4);
                z(i) = currentPos(3,4);
            end
            
            % Constraints (Absolute)
            x = max(model.xmin, min(model.xmax, x));
            y = max(model.ymin, min(model.ymax, y));
            z = max(model.zmin, min(model.zmax, z));
            
            position.x = x; position.y = y; position.z = z;
        end
    end
end

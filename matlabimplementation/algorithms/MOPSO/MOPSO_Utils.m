classdef MOPSO_Utils
    methods(Static)
        function b = Dominates(x, y)
            if isstruct(x)
                x = x.Cost;
            end
            if isstruct(y)
                y = y.Cost;
            end
            b = all(x <= y) && any(x < y) && all(isfinite(x));
        end

        function pop = DetermineDomination(pop)
            nPop = numel(pop);
            for i = 1:nPop
                pop(i).IsDominated = false;
            end
            for i = 1:nPop
                for j = 1:nPop
                    if i == j
                        continue;
                    end
                    if MOPSO_Utils.Dominates(pop(j), pop(i))
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
            dc = cmax - cmin;
            cmin = cmin - alpha * dc;
            cmax = cmax + alpha * dc;
            nObj = size(c, 1);
            empty_grid.LB = [];
            empty_grid.UB = [];
            Grid = repmat(empty_grid, nObj, 1);
            for j = 1:nObj
                cj = linspace(cmin(j), cmax(j), nGrid + 1);
                Grid(j).LB = [-inf, cj];
                Grid(j).UB = [cj, +inf];
            end
        end

        function particle = FindGridIndex(particle, Grid)
            nObj = numel(particle.Cost);
            nGrid = numel(Grid(1).LB) - 2;
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
                particle.GridIndex = (particle.GridIndex - 1) * nGrid + particle.GridSubIndex(j);
            end
        end

        function rep = DeleteOneRepMember(rep, gamma)
            GI = [rep.GridIndex];
            OC = unique(GI);
            N = zeros(size(OC));
            for k = 1:numel(OC)
                N(k) = sum(GI == OC(k));
            end
            P = exp(gamma * N);
            P = P / sum(P);
            sci = MOPSO_Utils.RouletteWheelSelection(P);
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
            P = exp(-beta * N);
            P = P / sum(P);
            sci = MOPSO_Utils.RouletteWheelSelection(P);
            sc = OC(sci);
            SCM = find(GI == sc);
            sm = SCM(randi([1 numel(SCM)]));
            leader = rep(sm);
        end
    end
end

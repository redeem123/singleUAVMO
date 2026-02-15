classdef UAV_MTOP < PROBLEM
% UAV_MTOP - Multitask multi-objective UAV path planning problem for PlatEMO.

    properties
        SubM;
        SubD;
        Models;
        NControl;
        Penalty = 1e6;
    end

    methods
        function Setting(obj)
            spec = obj.ParameterSet(struct());

            if ~isstruct(spec) || ~isfield(spec, 'models') || numel(spec.models) < 2
                error('UAV_MTOP requires parameter struct with at least two models in spec.models.');
            end

            obj.Models = spec.models;

            if isfield(spec, 'nControl') && isnumeric(spec.nControl) && isfinite(spec.nControl)
                obj.NControl = max(3, round(spec.nControl));
            else
                obj.NControl = max(3, round(obj.Models{1}.n));
            end

            varsPerTask = 3 * (obj.NControl - 2);
            taskCount = numel(obj.Models);

            obj.SubD = repmat(varsPerTask, 1, taskCount);
            obj.SubM = repmat(4, 1, taskCount);

            obj.M = 4;
            obj.D = varsPerTask + 1; % Last variable is skill factor (task id)

            obj.lower = [zeros(1, obj.D - 1), 1];
            obj.upper = [ones(1, obj.D - 1), taskCount];
            obj.encoding = [ones(1, obj.D - 1), 2];
        end

        function Population = Evaluation(obj, varargin)
            X = varargin{1};
            PopDec = max(min(X, repmat(obj.upper, size(X, 1), 1)), repmat(obj.lower, size(X, 1), 1));
            PopDec(:, end) = round(PopDec(:, end));

            PopObj = zeros(size(PopDec, 1), obj.M);
            PopCon = zeros(size(PopDec, 1), 1);

            for i = 1:size(PopDec, 1)
                taskId = min(max(PopDec(i, end), 1), numel(obj.Models));
                model = obj.Models{taskId};

                path = UAV_MTOP.DecodePath(PopDec(i, 1:end-1), model, obj.NControl);
                objs = evaluate_path(path, model);

                if any(~isfinite(objs))
                    PopObj(i, :) = obj.Penalty;
                    PopCon(i) = 1;
                else
                    PopObj(i, :) = objs;
                end
            end

            Population = SOLUTION(PopDec, PopObj, PopCon, varargin{2:end});
            obj.FE = obj.FE + length(Population);
        end

        function PopDec = CalDec(obj, PopDec)
            PopDec = max(min(PopDec, repmat(obj.upper, size(PopDec, 1), 1)), repmat(obj.lower, size(PopDec, 1), 1));
            PopDec(:, end) = round(PopDec(:, end));
        end

        function R = GetOptimum(obj, N)
            R = ones(N, obj.M);
        end

        function R = GetPF(~)
            R = [];
        end
    end

    methods(Static)
        function path = DecodePath(normDec, model, nControl)
            nControl = max(3, round(nControl));
            nMid = nControl - 2;
            needed = 3 * nMid;

            normDec = normDec(:)';
            if numel(normDec) < needed
                normDec = [normDec, 0.5 * ones(1, needed - numel(normDec))];
            elseif numel(normDec) > needed
                normDec = normDec(1:needed);
            end

            mid = reshape(normDec, [nMid, 3]);
            mid = max(0, min(1, mid));

            x = double(model.xmin) + mid(:, 1) * (double(model.xmax) - double(model.xmin));
            y = double(model.ymin) + mid(:, 2) * (double(model.ymax) - double(model.ymin));
            zAlpha = mid(:, 3);

            [x, idx] = sort(x, 'ascend');
            y = y(idx);
            zAlpha = zAlpha(idx);

            safeH = 0;
            if isfield(model, 'safeH') && ~isempty(model.safeH)
                safeH = double(model.safeH);
            end

            z = zeros(nMid, 1);
            for i = 1:nMid
                xi = max(1, min(size(model.H, 2), round(x(i))));
                yi = max(1, min(size(model.H, 1), round(y(i))));
                ground = double(model.H(yi, xi));

                minZ = max(double(model.zmin), ground + safeH);
                maxZ = double(model.zmax);
                if maxZ <= minZ
                    z(i) = minZ;
                else
                    z(i) = minZ + zAlpha(i) * (maxZ - minZ);
                end
            end

            path = zeros(nControl, 3);
            path(1, :) = double(model.start(:))';
            path(end, :) = double(model.end(:))';
            path(2:end-1, :) = [x, y, z];

            path(:, 1) = max(double(model.xmin), min(double(model.xmax), path(:, 1)));
            path(:, 2) = max(double(model.ymin), min(double(model.ymax), path(:, 2)));
            path(:, 3) = max(double(model.zmin), min(double(model.zmax), path(:, 3)));
        end
    end
end

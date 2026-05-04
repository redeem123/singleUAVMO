classdef PythonBridgeLegacyProblem < PROBLEM
% Legacy GLOBAL-compatible problem whose evaluation is delegated to Python.

    properties(Access = private)
        configPath
        workDir
        pythonExecutable
        pythonPath
        contextPath
        lastDec
        lastCon
    end

    methods
        function obj = PythonBridgeLegacyProblem()
            obj@PROBLEM();
        end

        function Setting(obj)
            obj.configPath = getenv('PYTHON_BRIDGE_CONFIG');
            cfg = load(obj.configPath);
            obj.M = double(cfg.M);
            obj.D = double(cfg.D);
            obj.lower = double(cfg.lower);
            obj.upper = double(cfg.upper);
            obj.encoding = ones(1,obj.D);
            obj.PF = zeros(10000,obj.M);
            Global = GLOBAL.GetObj();
            if ~isempty(Global)
                Global.M = obj.M;
                Global.D = obj.D;
                Global.lower = obj.lower;
                Global.upper = obj.upper;
                Global.encoding = 'real';
            end
            obj.workDir = char(cfg.workDir);
            obj.pythonExecutable = char(cfg.pythonExecutable);
            obj.pythonPath = char(cfg.pythonPath);
            obj.contextPath = char(cfg.contextPath);
        end

        function PopDec = Init(obj,N)
            N = double(N);
            PopDec = unifrnd(repmat(obj.lower,N,1),repmat(obj.upper,N,1));
        end

        function PopObj = CalObj(obj,PopDec)
            [PopObj, PopCon] = obj.EvaluateInPython(PopDec);
            obj.lastDec = PopDec;
            obj.lastCon = PopCon;
        end

        function PopCon = CalCon(obj,PopDec)
            if ~isempty(obj.lastDec) && isequal(size(obj.lastDec),size(PopDec)) && isequal(obj.lastDec,PopDec)
                PopCon = obj.lastCon;
            else
                [~, PopCon] = obj.EvaluateInPython(PopDec);
            end
        end

        function PF = PF(obj,N)
            if nargin < 2
                N = 10000;
            end
            PF = zeros(double(N),obj.M);
        end

        function R = GetPF(obj)
            R = zeros(10000,obj.M);
        end
    end

    methods(Access = private)
        function [PopObj, PopCon] = EvaluateInPython(obj,PopDec)
            cfg = struct( ...
                'workDir',obj.workDir, ...
                'pythonExecutable',obj.pythonExecutable, ...
                'pythonPath',obj.pythonPath, ...
                'contextPath',obj.contextPath);
            request = struct('PopDec',double(PopDec));
            [PopObj, PopCon] = BridgeEvaluateInPython(cfg,request);
            if size(PopObj,2) > obj.M
                PopObj = PopObj(:,1:obj.M);
            end
        end
    end
end

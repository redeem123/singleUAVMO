classdef PythonBridgeProblem < PROBLEM
% Generic PlatEMO problem whose evaluation is delegated to Python.
% This is not a UAV problem definition; it only forwards decision matrices.

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
        function obj = PythonBridgeProblem(varargin)
            obj@PROBLEM(varargin{:});
        end

        function Setting(obj)
            obj.configPath = obj.parameter{1};
            cfg = load(obj.configPath);
            obj.M      = double(cfg.M);
            obj.D      = double(cfg.D);
            obj.N      = double(cfg.N);
            obj.maxFE  = double(cfg.maxFE);
            obj.lower  = double(cfg.lower);
            obj.upper  = double(cfg.upper);
            obj.encoding = ones(1,obj.D);
            obj.workDir = char(cfg.workDir);
            obj.pythonExecutable = char(cfg.pythonExecutable);
            obj.pythonPath = char(cfg.pythonPath);
            obj.contextPath = char(cfg.contextPath);
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
        end
    end
end

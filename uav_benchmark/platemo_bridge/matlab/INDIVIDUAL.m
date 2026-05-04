classdef INDIVIDUAL < handle
% Minimal legacy PlatEMO individual adapter for reference algorithms.

    properties(SetAccess = private)
        dec
        obj
        con
        add
    end
    properties(Dependent)
        decs
        objs
        cons
    end

    methods
        function obj = INDIVIDUAL(PopDec,varargin)
            if nargin == 0
                return;
            end
            PopDec = double(PopDec);
            Global = GLOBAL.GetObj();
            PopObj = Global.problem.CalObj(PopDec);
            PopCon = Global.problem.CalCon(PopDec);
            if isempty(PopCon)
                PopCon = zeros(size(PopDec,1),1);
            end
            obj(1,size(PopDec,1)) = INDIVIDUAL;
            for i = 1 : size(PopDec,1)
                obj(i).dec = PopDec(i,:);
                obj(i).obj = PopObj(i,:);
                obj(i).con = PopCon(i,:);
                obj(i).add = cellfun(@(A) A(i,:),varargin,'UniformOutput',false);
            end
            Global.evaluated = Global.evaluated + size(PopDec,1);
        end

        function value = get.decs(obj)
            value = cat(1,obj.dec);
        end

        function value = get.objs(obj)
            value = cat(1,obj.obj);
        end

        function value = get.cons(obj)
            value = cat(1,obj.con);
        end

        function value = adds(obj,default)
            if nargin < 2
                default = [];
            end
            if isempty(obj) || isempty(obj(1).add)
                value = default;
                return;
            end
            value = cat(1,obj.add{:});
        end

        function varargout = subsref(obj,S)
            if strcmp(S(1).type,'()') && numel(S) >= 2 && strcmp(S(2).type,'.') && any(strcmp(S(2).subs,{'dec','obj','con','decs','objs','cons','adds'}))
                subset = builtin('subsref',obj,S(1));
                [varargout{1:nargout}] = subsref(subset,S(2:end));
            elseif strcmp(S(1).type,'.') && any(strcmp(S(1).subs,{'dec','obj','con','decs','objs','cons','adds'}))
                switch S(1).subs
                    case {'dec','decs'}
                        value = cat(1,obj.dec);
                    case {'obj','objs'}
                        value = cat(1,obj.obj);
                    case {'con','cons'}
                        value = cat(1,obj.con);
                    case 'adds'
                        if numel(S) >= 2 && strcmp(S(2).type,'()')
                            value = adds(obj,S(2).subs{:});
                            S(2) = [];
                        else
                            value = adds(obj,[]);
                        end
                end
                if numel(S) > 1
                    value = builtin('subsref',value,S(2:end));
                end
                varargout{1} = value;
            else
                [varargout{1:nargout}] = builtin('subsref',obj,S);
            end
        end

        function n = numArgumentsFromSubscript(obj,S,indexingContext) %#ok<INUSD>
            if ~isempty(S) && strcmp(S(1).type,'()') && numel(S) >= 2 && strcmp(S(2).type,'.') && any(strcmp(S(2).subs,{'dec','obj','con','decs','objs','cons','adds'}))
                n = 1;
            elseif ~isempty(S) && strcmp(S(1).type,'.') && any(strcmp(S(1).subs,{'dec','obj','con','decs','objs','cons','adds'}))
                n = 1;
            else
                n = builtin('numArgumentsFromSubscript',obj,S,indexingContext);
            end
        end
    end
end

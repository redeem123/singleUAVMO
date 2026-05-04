function [PopObj, PopCon] = BridgeEvaluateInPython(configOrPath, requestVars)
% Evaluate MATLAB candidates with the Python UAV benchmark objective.
%
% MATLAB algorithms keep ownership of initialization, variation, repair,
% selection, and archiving. This helper is the single boundary where candidate
% decisions or paths are scored by the repository's Python benchmark evaluator.

    if ischar(configOrPath) || isstring(configOrPath)
        cfg = load(char(configOrPath));
    else
        cfg = configOrPath;
    end

    requestPath = [tempname(char(cfg.workDir)), '_request.mat'];
    responsePath = [tempname(char(cfg.workDir)), '_response.mat'];
    cleanupRequest = onCleanup(@() DeleteIfExists(requestPath)); %#ok<NASGU>
    cleanupResponse = onCleanup(@() DeleteIfExists(responsePath)); %#ok<NASGU>

    save(requestPath, '-struct', 'requestVars');
    setenv('PYTHONPATH', char(cfg.pythonPath));
    cmd = sprintf('"%s" -m uav_benchmark.platemo_bridge.evaluator --context "%s" --request "%s" --response "%s"', ...
        char(cfg.pythonExecutable), char(cfg.contextPath), requestPath, responsePath);
    [status, output] = system(cmd);
    if status ~= 0
        error('BridgeEvaluateInPython:PythonEvaluationFailed','%s',output);
    end

    payload = load(responsePath);
    PopObj = double(payload.PopObj);
    if isfield(payload,'PopCon')
        PopCon = double(payload.PopCon);
    else
        PopCon = zeros(size(PopObj,1),1);
    end
end

function DeleteIfExists(path)
    if exist(path,'file') == 2
        delete(path);
    end
end

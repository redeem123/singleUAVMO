% Script to aggregate benchmark results
clear all; clc;
resultsDir = '.'; % examples/NSGA-II
d = dir(fullfile(resultsDir, 'Population_*'));
folders = {d([d.isdir]).name};

fprintf('Problem | Mean HV | Std HV | Mean PD | Std PD\n');
fprintf('---|---|---|---|---\n');

for i = 1:length(folders)
    folder = folders{i};
    matFile = fullfile(resultsDir, folder, 'final_hv.mat');
    if exist(matFile, 'file')
        data = load(matFile);
        % data.bestScores is Runs x 2
        hv = data.bestScores(:,1);
        pd = data.bestScores(:,2);
        
        problemName = strrep(folder, 'Population_', '');
        fprintf('%s | %.2f | %.2f | %.2f | %.2f\n', ...
            problemName, mean(hv), std(hv), mean(pd), std(pd));
    end
end


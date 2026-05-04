function [Population,W] = updateWeight(Population,W,Z,EP,nus,M)
% Objective-count-generic version of the author/PlatEMO updateWeight helper.

    if nargin < 6
        sample = [Population.objs];
        M = length(sample) / max(1,length(Population));
    end
    M = double(M);
    obj = [Population.objs];
    PopObj = reshape(obj,M,length(Population))';
    [N,~] = size(PopObj);

    Combine = [Population,EP];
    com_obj = [Combine.objs];
    ComObj = reshape(com_obj,M,length(Combine))';
    CombineObj = abs(ComObj-repmat(Z,length(Combine),1));
    g = zeros(length(Combine),size(W,1));
    for i = 1 : size(W,1)
        g(:,i) = max(CombineObj.*repmat(W(i,:),length(Combine),1),[],2);
    end
    [~,best] = min(g,[],1);
    Population = Combine(best);

    Dis = pdist2(PopObj,PopObj);
    Dis(logical(eye(length(Dis)))) = inf;
    Del = false(1,length(Population));
    while sum(Del) < min(nus,length(EP))
        Remain = find(~Del);
        if numel(Remain) <= 1
            break;
        end
        subDis = sort(Dis(Remain,Remain),2);
        [~,worst] = min(prod(subDis(:,1:min(M,length(Remain))),2));
        Del(Remain(worst)) = true;
    end
    Population = Population(~Del);
    W = W(~Del,:);

    Combine = [Population,EP];
    Selected = false(1,length(Combine));
    Selected(1:length(Population)) = true;
    com_obj = [Combine.objs];
    ComObj = reshape(com_obj,M,length(Combine))';
    Dis = pdist2(ComObj,ComObj);
    Dis(logical(eye(length(Dis)))) = inf;
    while sum(Selected) < min(N,length(Selected))
        subDis = sort(Dis(~Selected,Selected),2);
        if isempty(subDis)
            break;
        end
        [~,best] = max(prod(subDis(:,1:min(M,size(subDis,2))),2));
        Remain = find(~Selected);
        Selected(Remain(best)) = true;
    end

    picked = Selected(length(Population)+1:end);
    if any(picked)
        ep_obj = [EP(picked).objs];
        newObjs = reshape(ep_obj,M,length(EP(picked)))';
        denom = max(abs(newObjs-repmat(Z,size(newObjs,1),1)),1e-12);
        temp = 1./denom;
        W = [W;temp./repmat(sum(temp,2),1,size(temp,2))];
    end
    Population = Combine(Selected);
end

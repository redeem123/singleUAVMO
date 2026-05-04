function Offspring = GA(Population)
% Compatibility GA operator for EMMOP's old PlatEMO-style ActionSelection.

    Global = GLOBAL.GetObj();
    Parent = Population.decs;
    [N,D] = size(Parent);
    if mod(N,2) == 1
        Parent = [Parent; Parent(randi(N),:)];
        N = N + 1;
    end
    Parent1 = Parent(1:floor(N/2),:);
    Parent2 = Parent(floor(N/2)+1:N,:);
    proC = 1;
    disC = 20;
    proM = 1;
    disM = 20;

    beta = zeros(size(Parent1));
    mu = rand(size(Parent1));
    beta(mu<=0.5) = (2*mu(mu<=0.5)).^(1/(disC+1));
    beta(mu>0.5) = (2-2*mu(mu>0.5)).^(-1/(disC+1));
    beta = beta.*(-1).^randi([0,1],size(beta));
    beta(rand(size(beta))<0.5) = 1;
    beta(repmat(rand(size(Parent1,1),1)>proC,1,D)) = 1;
    OffDec = [(Parent1+Parent2)/2+beta.*(Parent1-Parent2)/2
              (Parent1+Parent2)/2-beta.*(Parent1-Parent2)/2];

    Lower = repmat(Global.lower,size(OffDec,1),1);
    Upper = repmat(Global.upper,size(OffDec,1),1);
    Site = rand(size(OffDec)) < proM/D;
    mu = rand(size(OffDec));
    temp = Site & mu<=0.5;
    OffDec = min(max(OffDec,Lower),Upper);
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
        (1-(OffDec(temp)-Lower(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu>0.5;
    OffDec(temp) = OffDec(temp)+(Upper(temp)-Lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
        (1-(Upper(temp)-OffDec(temp))./(Upper(temp)-Lower(temp))).^(disM+1)).^(1/(disM+1)));
    OffDec = min(max(OffDec,Lower),Upper);
    Offspring = INDIVIDUAL(OffDec(1:length(Population),:));
end
